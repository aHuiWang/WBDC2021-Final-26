# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 18:07
# @Author  : Hui Wang


import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from src.model.base_model import ContextRecommender
from src.model.layers import MLPLayers, RegLoss, MOE


class DCN(ContextRecommender):
    """Deep & Cross Network replaces the wide part in Wide&Deep with cross network,
    automatically construct limited high-degree cross features, and learns the corresponding weights.
    """
    def __init__(self, args, field_names, field_sources, field2type, field2num):
        super(DCN, self).__init__(args, field_names, field_sources, field2type, field2num)

        # load parameters info
        self.cross_layer_num = args.cross_layer_num
        self.mlp_hidden_size = args.mlp_hidden_size
        self.dropout_prob = args.dropout_prob
        self.moe_dropout_prob = args.moe_dropout_prob
        # self.reg_weight = args.reg_weight

        self.reg_weight = 1.0

        # define layers and loss
        # init weight and bias of each cross layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(torch.randn(self.num_feature_field * self.embedding_size).to(self.device))
            for _ in range(self.cross_layer_num)
        )
        self.cross_layer_b = nn.ParameterList(
            nn.Parameter(torch.zeros(self.num_feature_field * self.embedding_size).to(self.device))
            for _ in range(self.cross_layer_num)
        )

        # size of mlp hidden layer
        size_list = [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size
        # size of cross network output
        in_feature_num = self.embedding_size * self.num_feature_field + self.mlp_hidden_size[-1]

        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob, bn=True)
        self.moe = MOE(feature_size=in_feature_num, 
                       num_classes=7, dropout_prob=self.moe_dropout_prob)
        self.reg_loss = RegLoss()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.
        .. math:: x_{l+1} = x_0 {x_l^T} w_l + b_l + x_l
        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`w_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.
        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.
        Returns:
            torch.Tensor:output of cross network, [batch_size, num_feature_field * embedding_size]
        """
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.tensordot(x_l, self.cross_layer_w[i], dims=([1], [0]))
            xl_dot = (x_0.transpose(0, 1) * xl_w).transpose(0, 1)
            x_l = xl_dot + self.cross_layer_b[i] + x_l
        return x_l

    def forward(self, interaction):
        dcn_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        batch_size = dcn_all_embeddings.shape[0]
        dcn_all_embeddings = dcn_all_embeddings.view(batch_size, -1)

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        
        logits = self.moe(stack)

        return logits

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return output

    def predict(self, interaction):
        return self.forward(interaction)