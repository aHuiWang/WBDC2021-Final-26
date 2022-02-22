# -*- coding: utf-8 -*-
# @Time    : 2021/6/23 0:11
# @Author  : Hui Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from src.model.dcn_multi import DCN
from src.model.pnn_multi import PNN


class StackModel(nn.Module):
    """ AutoInt is a novel CTR prediction model based on self-attention mechanism,
    which can automatically learn high-order feature interactions in an explicit fashion.
    """

    def __init__(self, args, field_names1, field_names2, field_names3, field_names4,
                 field_sources, field2type, field2num):
        super(StackModel, self).__init__()
        self.dcn1 = DCN(args, field_names1, field_sources, field2type, field2num)
        self.dcn2 = DCN(args, field_names2, field_sources, field2type, field2num)
        self.dcn3 = DCN(args, field_names3, field_sources, field2type, field2num)
        self.pnn = PNN(args, field_names3, field_sources, field2type, field2num)
        
        args.mlp_hidden_size=[256]
        args.cross_layer_num=1
        self.dcn1_ = DCN(args, field_names1, field_sources, field2type, field2num)
        self.dcn2_ = DCN(args, field_names2, field_sources, field2type, field2num)
        self.dcn3_ = DCN(args, field_names3, field_sources, field2type, field2num)
        self.dcn4_ = DCN(args, field_names4, field_sources, field2type, field2num)
        self.pnn_ = PNN(args, field_names4, field_sources, field2type, field2num)

    def forward(self, interaction):
        dcn_output1 = self.dcn1(interaction)
        dcn_output2 = self.dcn2(interaction)
        dcn_output3 = self.dcn3(interaction)
        pnn_output = self.pnn(interaction)
        dcn_output1_ = self.dcn1_(interaction)
        dcn_output2_ = self.dcn2_(interaction)
        dcn_output3_ = self.dcn3_(interaction)
        dcn_output4_ = self.dcn4_(interaction)
        pnn_output_ = self.pnn_(interaction)
        output = (dcn_output1 + dcn_output2 + dcn_output3 + pnn_output + dcn_output1_ + dcn_output2_ + dcn_output3_ + dcn_output4_ + pnn_output_) / 9
        return torch.sigmoid(output)

    def calculate_loss(self, interaction):
        return self.forward(interaction)

    def predict(self, interaction):
        return self.forward(interaction)