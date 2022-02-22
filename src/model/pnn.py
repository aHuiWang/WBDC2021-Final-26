# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 1:52
# @Author  : Hui Wang



import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_

from src.model.base_model import ContextRecommender
from src.model.layers import MLPLayers


class PNN(ContextRecommender):
    """PNN calculate inner and outer product of feature embedding.
    You can choose the product option with the parameter of use_inner and use_outer
    """

    def __init__(self, args, field_names, field_sources, field2type, field2num):
        super(PNN, self).__init__(args, field_names, field_sources, field2type, field2num)

        # load parameters info
        self.mlp_hidden_size = args.mlp_hidden_size
        self.dropout_prob = args.dropout_prob
        # self.use_inner = args.use_inner
        # self.use_outer = args.use_outer
        # self.reg_weight = args.reg_weight
        if args.model_name == 'PNN':
            self.use_inner = False
            self.use_outer = True
        elif args.model_name == 'PNN1':
            self.use_inner = True
            self.use_outer = False
        elif args.model_name == 'PNN2':
            self.use_inner = True
            self.use_outer = True
        self.reg_weight = args.reg_weight

        self.num_pair = int(self.num_feature_field * (self.num_feature_field - 1) / 2)

        # define layers and loss
        product_out_dim = self.num_feature_field * self.embedding_size
        if self.use_inner:
            product_out_dim += self.num_pair
            self.inner_product = InnerProductLayer(self.num_feature_field, device=self.device)

        if self.use_outer:
            product_out_dim += self.num_pair
            self.outer_product = OuterProductLayer(self.num_feature_field, self.embedding_size, device=self.device)
        size_list = [product_out_dim] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob, bn=False)
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def reg_loss(self):
        """Calculate the L2 normalization loss of model parameters.
        Including weight matrices of mlp layers.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.reg_weight * parm.norm(2)
        return reg_loss

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, interaction):
        pnn_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        batch_size = pnn_all_embeddings.shape[0]
        # linear part
        linear_part = pnn_all_embeddings.view(batch_size, -1)  # [batch_size,num_field*embed_dim]
        output = [linear_part]
        # second order part
        if self.use_inner:
            inner_product = self.inner_product(pnn_all_embeddings).view(batch_size, -1)  # [batch_size,num_pairs]
            output.append(inner_product)
        if self.use_outer:
            outer_product = self.outer_product(pnn_all_embeddings).view(batch_size, -1)  # [batch_size,num_pairs]
            output.append(outer_product)
        output = torch.cat(output, dim=1)  # [batch_size,d]

        output = self.predict_layer(self.mlp_layers(output))  # [batch_size,1]
        output = self.sigmoid(output)
        return output.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)

        return output, self.loss(output, label) + self.reg_loss()

    def predict(self, interaction):
        return self.forward(interaction)


class InnerProductLayer(nn.Module):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.
    """

    def __init__(self, num_feature_field, device):
        """
        Args:
            num_feature_field(int) :number of feature fields.
            device(torch.device) : device object of the model.
        """
        super(InnerProductLayer, self).__init__()
        self.num_feature_field = num_feature_field
        self.to(device)

    def forward(self, feat_emb):
        """
        Args:
            feat_emb(torch.FloatTensor) :3D tensor with shape: [batch_size,num_pairs,embedding_size].
        Returns:
            inner_product(torch.FloatTensor): The inner product of input tensor. shape of [batch_size, num_pairs]
        """
        # num_pairs = num_feature_field * (num_feature_field-1) / 2
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]  # [batch_size, num_pairs, emb_dim]
        q = feat_emb[:, col]  # [batch_size, num_pairs, emb_dim]

        inner_product = p * q

        return inner_product.sum(dim=-1)  # [batch_size, num_pairs]


class OuterProductLayer(nn.Module):
    """OuterProduct Layer used in PNN. This implementation is
    adapted from code that the author of the paper published on https://github.com/Atomu2014/product-nets.
    """

    def __init__(self, num_feature_field, embedding_size, device):
        """
        Args:
            num_feature_field(int) :number of feature fields.
            embedding_size(int) :number of embedding size.
            device(torch.device) : device object of the model.
        """
        super(OuterProductLayer, self).__init__()

        self.num_feature_field = num_feature_field
        num_pairs = int(num_feature_field * (num_feature_field - 1) / 2)
        embed_size = embedding_size

        self.kernel = nn.Parameter(torch.rand(embed_size, num_pairs, embed_size), requires_grad=True)
        nn.init.xavier_uniform_(self.kernel)

        self.to(device)

    def forward(self, feat_emb):
        """
        Args:
            feat_emb(torch.FloatTensor) :3D tensor with shape: [batch_size,num_pairs,embedding_size].
        Returns:
            outer_product(torch.FloatTensor): The outer product of input tensor. shape of [batch_size, num_pairs]
        """
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]  # [batch_size, num_pairs, emb_dim]
        q = feat_emb[:, col]  # [batch_size, num_pairs, emb_dim]

        # -------------------------

        p.unsqueeze_(dim=1)  # [batch_size, 1, num_pairs, emb_dim]

        p = torch.mul(p, self.kernel.unsqueeze(0))  # [batch_size,emb_dim,num_pairs,emb_dim]
        p = torch.sum(p, dim=-1)  # [batch_size,emb_dim,num_pairs]
        p = torch.transpose(p, 2, 1)  # [batch_size,num_pairs,emb_dim]

        outer_product = p * q
        return outer_product.sum(dim=-1)  # [batch_size,num_pairs]