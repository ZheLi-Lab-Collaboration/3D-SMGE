# -*- coding: utf-8 -*-
# @Time    : 2022/3/20 11:48
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com


import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from caSchnetpack.activationFunction.dy_relu import DyReLUB
from caSchnetpack.activationFunction.acf_gelus import gelu_bert, soft_plus, gelus_gt2


class atomEF(nn.Module):
    def __init__(
            self,
            n_in,
            n_out,
            vocab_size=100,
            embedding_size=128
    ):
        super(atomEF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=3, padding=1)
        self.dy_relu = DyReLUB(n_out, conv_type="1d")
        self.gelu = gelus_gt2()
        self.norm = nn.BatchNorm1d(n_out)
        # self.linear = nn.Linear(n_out, n_in, bias=True)


    def forward(self, x):
        # embedding_x = self.embedding(x)
        # print("embedding_x.size()", embedding_x.size())
        # embedding_x = embedding_x.permute(0, 2, 1)
        # conv1_x = self.conv1d(embedding_x)
        x = x.permute(0, 2, 1)
        conv1_x = self.conv1d(x)
        y_ = self.dy_relu(self.norm(conv1_x))
        # print("y_.size()", y_.size())
        y = y_.permute(0, 2, 1)
        # y = self.linear(y_)
        # y = y.permute(0, 2, 1)
        # print("y.size()", y.size())
        return y

#
# if __name__ == "__main__":
#     model = atomEF(n_in=128, n_out=128)
#     print(model)
#     x_np = np.arange(0, 100).reshape(5, 20)
#
#     x = torch.LongTensor(x_np)
#     # x = torch.LongTensor([[0, 2, 0, 1],
#     #                       [1, 3, 4, 4]])
#
#     # a = torch.nn.Embedding(100, 128, padding_idx=0)
#     # b = a(x)
#     # print(b)
#
#     print(x)
#     y = model(x)
#
#     print(y)
#     print(y.size())
#     # print(y.size())
#     # print(y.unsqueeze(-1))
#     # print((y.unsqueeze(-1)).size())

