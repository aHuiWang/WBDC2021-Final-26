# -*- coding: utf-8 -*-
# @Time    : 2021/6/13 20:16
# @Author  : Hui Wang

import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from src.model.base_model import ContextRecommender
from src.model.layers import BaseFactorizationMachine, MLPLayers


class DeepFM(ContextRecommender):
    """DeepFM is a DNN enhanced FM which both use a DNN and a FM to calculate feature interaction.
    Also DeepFM can be seen as a combination of FNN and FM.
    """

    def __init__(self, args, field_names, field_sources, field2type, field2num):
        super(DeepFM, self).__init__(args, field_names, field_sources, field2type, field2num)

        # load parameters info
        self.mlp_hidden_size = args.mlp_hidden_size
        self.dropout_prob = args.dropout_prob

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)  # Linear product to the final score
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

    def forward(self, interaction):
        deepfm_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.first_order_linear(interaction) + self.fm(deepfm_all_embeddings)

        y_deep = self.deep_predict_layer(self.mlp_layers(deepfm_all_embeddings.view(batch_size, -1)))
        y = self.sigmoid(y_fm + y_deep)
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return output, self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)