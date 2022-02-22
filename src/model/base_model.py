# -*- coding: utf-8 -*-
# @Time    : 2021/6/13 20:15
# @Author  : Hui Wang

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from src.model.layers import FMEmbedding, MLPLayers

from src.utils import kmax_pooling

class ContextRecommender(nn.Module):
    """This is a abstract context-aware recommender. All the context-aware model should implement this class.
    The base context-aware recommender class provide the basic embedding function of feature fields which also
    contains a first-order part of feature fields.
    """

    def __init__(self, args, field_names, field_sources, field2type, field2num):
        super(ContextRecommender, self).__init__()
        self.logger = getLogger()

        self.device = args.device
        self.field_names = field_names
        self.field_sources = field_sources
        self.field2type = field2type
        self.field2num = field2num

        self.LABEL = args.label
        self.embedding_size = args.embedding_size
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.token_seq_field_names_char = []
        self.token_seq_field_names_word = []
        self.token_seq_field_names_pre = []
        self.feed_seq_field_name = []
        self.num_feature_field = 0

        for field_name in self.field_names:
            if 'his_seq' in field_name:
                self.feed_seq_field_name.append(field_name)
            
            elif 'char' in field_name:
                self.token_seq_field_names_char.append(field_name)
                
            else:
                feature_type = self.field2type.get(field_name, 'token')
                field_dim = self.field2num.get(field_name, 4)
                if feature_type == 'token':
                    self.token_field_names.append(field_name)
                    self.token_field_dims.append(field_dim)
                elif feature_type == 'token_seq':
                    self.token_seq_field_names.append(field_name)
                    self.token_seq_field_dims.append(field_dim)
                else:
                    #
                    self.float_field_names.append(field_name)
                    self.float_field_dims.append(1)
            self.num_feature_field += 1
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, self.embedding_size
            )
        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(
                np.sum(self.float_field_dims, dtype=np.int32), self.embedding_size
            )
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, self.embedding_size, padding_idx=0))

        if len(self.token_seq_field_names_char) > 0:
            self.char_embeddings = nn.Embedding(33379, args.w2v_emb, padding_idx=0)
            self.token_seq_gru_char = nn.ModuleList()
            for _ in self.token_seq_field_names_char:
                self.token_seq_gru_char.append(
                    nn.LSTM(
                        input_size=args.w2v_emb,
                        hidden_size=args.embedding_size,
                        num_layers=2,
                        bias=False,
                        batch_first=True,
                        bidirectional=False))
        if len(self.feed_seq_field_name) > 0:
            self.feed_seq_grus = nn.ModuleList()
            for _ in self.feed_seq_field_name:
                self.feed_seq_grus.append(
                    nn.LSTM(
                        input_size=args.embedding_size,
                        hidden_size=args.embedding_size,
                        num_layers=2,
                        bias=False,
                        batch_first=True,
                        bidirectional=False
                ))

    def embed_float_fields(self, float_fields, embed=True):
        """Embed the float feature columns
        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field]
            embed (bool): Return the embedding of columns or just the columns itself. default=True
        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[1]
        # [batch_size, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        return float_embedding

    def embed_token_fields(self, token_fields):
        """Embed the token feature columns
        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]
        Returns:
            torch.FloatTensor: The result embedding tensor of token columns.
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode='mean'):
        """Embed the token feature columns
        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean
        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(token_seq_embedding)  # [batch_size, seq_len, embed_dim]
            if mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]
                result = torch.max(masked_token_seq_embedding, dim=1, keepdim=True)[0]  # [batch_size, 1, embed_dim]
            elif mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1)  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=1)  # [batch_size, num_token_seq_field, embed_dim]


    def embed_token_seq_fields_char(self, token_seq_fields_char):
        """Embed the token feature columns
        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean
        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields_char):

            token_seq_embedding = self.char_embeddings(token_seq_field)  # [batch_size, seq_len, embed_dim]
            gru_layers = self.token_seq_gru_char[i]
            mask = (token_seq_field != 0).float()  # [batch_size, seq_len]
            gru_encoded, hn = gru_layers(token_seq_embedding)
            # gru_encoded = token_seq_embedding
            mask = mask.unsqueeze(2).expand_as(gru_encoded)  # [batch_size, seq_len, embed_dim]

            masked_gru_output = gru_encoded - (1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]

            result = kmax_pooling(masked_gru_output, dim=1, k=1)
            # result = torch.max(masked_gru_output, dim=1, keepdim=True)[0]  # [batch_size, 1, embed_dim]
            
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=1)  # [batch_size, num_token_seq_field, embed_dim]
    
    def embed_feed_seq(self, gru_layer, feed_seq, att_layer=None, queries=None):
        """Embed the token feature columns
        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean
        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        
        feed_seq_embedding = self.token_embedding_table.embedding(feed_seq)  # [batch_size, seq_len, embed_dim]
            
        mask = (feed_seq != 0).float()  # [batch_size, seq_len]
        if gru_layer:
            gru_encoded, hn = gru_layer(feed_seq_embedding)
        else:
            gru_encoded = feed_seq_embedding
        if att_layer:
            keys = gru_encoded
            embedding_size = queries.shape[-1]  # H
            hist_len = keys.shape[1]  # T
            queries = queries.repeat(1, hist_len)

            queries = queries.view(-1, hist_len, embedding_size)

            # MLP Layer
            input_tensor = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
            output = att_layer(input_tensor) # [B L 1]
            output = output / (embedding_size ** 0.5)

            mask = mask.unsqueeze(2).expand_as(output)  # [batch_size, seq_len, embed_dim]
            output = output * mask
            result = torch.matmul(output.transpose(-1, -2), keys) # [B 1 H]
        else:
            
            mask = mask.unsqueeze(2).expand_as(gru_encoded)  # [batch_size, seq_len, embed_dim]
            masked_gru_output = gru_encoded - (1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]
            result = kmax_pooling(masked_gru_output, dim=1, k=1)
            result = torch.max(masked_gru_output, dim=1, keepdim=True)[0]  # [batch_size, 1, embed_dim]

        fields_result.append(result)
        
        return torch.cat(fields_result, dim=1)  # [batch_size, num_token_seq_field, embed_dim]

    def concat_embed_input_fields(self, interaction):
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        return torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.
        Args:
            interaction (Interaction): The input data collection.
        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 2:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)  # [batch_size, num_float_field]
        else:
            float_fields = None
        # [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields, mode='max')

        token_seq_fields_char = []
        for field_name in self.token_seq_field_names_char:
            token_seq_fields_char.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None

        # TODO 特殊处理文本描述
        if token_seq_fields_char:
            token_seq_fields_embedding_char = self.embed_token_seq_fields_char(token_seq_fields_char)
            if token_seq_fields_embedding is None:
                token_seq_fields_embedding = token_seq_fields_embedding_char
            else:
                token_seq_fields_embedding = torch.cat([token_seq_fields_embedding, token_seq_fields_embedding_char],
                                                   dim=1)
                
        if self.feed_seq_field_name:
            for i, field_name in enumerate(self.feed_seq_field_name):
                feed_seq = interaction[field_name]
                if 'neg' in field_name:
                    feed_seq_embeddings = self.embed_feed_seq(gru_layer=None, feed_seq=feed_seq)
                else:
                    feed_seq_embeddings = self.embed_feed_seq(gru_layer=self.feed_seq_grus[i], feed_seq=feed_seq)
                token_seq_fields_embedding = torch.cat([token_seq_fields_embedding, feed_seq_embeddings], dim=1)
        
        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat([token_fields_embedding, token_seq_fields_embedding], dim=1)

        dense_embedding = float_fields_embedding

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding