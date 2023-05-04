# -*- coding: utf-8 -*-
# @Time    : 2022/3/20 16:43
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

from caSchnetpack.caSchnet import caSchNet
import schnetpack as spk
import torch
import torch.nn as nn

#
#
# if __name__ == '__main__':
#     model = caSchNet(n_atom_basis=128,
#                                   n_filters=128,
#                                   n_interactions=5,
#                                   cutoff=10,
#                                   n_gaussians=25,
#                                   max_z=100)
#
#     # representation = \
#     #     spk.representation.SchNet(n_atom_basis=128,
#     #                               n_filters=128,
#     #                               n_interactions=5,
#     #                               cutoff=10,
#     #                               n_gaussians=25,
#     #                               max_z=100)
#     print(model)




from schnetpack.nn import Dense
import torch
import torch.nn as nn
#
# from schnetpack.nn.activations import shifted_softplus
#
# from caSchnetpack.activationFunction.dy_relu import DyReLUA, DyReLUB
#
# class test(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,):
#         super(test, self).__init__()
#
#         # self.dense = Dense(in_channels, out_channels, bias=True, activation=shifted_softplus)
#         self.linear = nn.Linear(in_channels, out_channels, bias=True)
#         self.relu = shifted_softplus
#
#
#     def forward(self, x):
#         # y1 = self.dense(x)
#         # x = x.permute(0, 2, 1)
#         x = self.linear(x)
#
#         y2 = self.relu(x)
#         # print(y1.size())
#         print(y2.size())
#         return y2
#
#
#
#
# if __name__ == '__main__':
#     x = torch.randn(2, 3, 128)
#     model = test(in_channels=128, out_channels=128)
#     y2 = model(x)
#     print(y2)


# if __name__ == '__main__':
#     conv1 = nn.Conv1d(in_channels=256, out_channels = 100, kernel_size = 2)
#     input = torch.randn(32, 35, 256, 256)
#     # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
#     # input = input.permute(0, 2, 1)
#     out = conv1(input)
#     print(out.size())

import numpy as np

if __name__ == '__main__':
    # a = torch.randn(25, 82, 63, 128)
    # d = torch.LongTensor(np.random.randn(25, 82))
    # model = nn.Embedding(100, 128)
    # y = model(d)
    # print(y)
    # print(y.size())

    # model = nn.MaxPool2d(kernel_size=5, stride=1, padding=1)
    # b = a.unsqueeze(-1)
    # c = a.reshape(-1)
    # print(c)
    # print(c.size())
    # print(b.size())
    # print(b)
    # y = model(a)
    # print(y)
    # print(y.size())
    # b = a[:, :, :, :, None]
    # print(b)
    # print(b.size())

    """cut_off测试，在小于5的地方，变为1，其他为0"""
    # a = torch.arange(0, 60).view(3, 4, 5)
    # print(a)
    # mask = (a <= 5).float()
    # mask = mask.unsqueeze(-1)
    # print(mask)
    # print(mask.size())

    """测试四维矩阵相乘（点乘）"""
    # a = torch.randn(2, 3)
    # print(a)
    # b = torch.randn(1, 3)
    # print(b)
    # c = a * b
    # print(c)
    # print(c.size())

    """测试输入特征为维度reshape"""
    # a = torch.randn(25, 82, 81, 25)
    # b = a.reshape(2050, -1, 25)
    # print(a)
    # print(b)
    # print(b.size())

    """测试caFilter"""
    # conv1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
    # input = torch.randn(32, 35, 256)
    # # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
    # input = input.permute(0, 2, 1)
    # out = conv1(input)
    # out = out.permute(0, 2, 1)
    # print(out.size())

    """测试caFilter"""
    # from schnetpack.nn.activations import shifted_softplus
    # filter_network = nn.Sequential(
    #     Dense(25, 128, activation=shifted_softplus),
    #     Dense(128, 128),
    # )
    # a = torch.randn(100, 50, 25)
    # y = filter_network(a)
    # print(y.size())

    # """测试三维变四维"""
    # a = torch.randn(25, 81, 82, 128)
    # a_size = a.size()
    # b = a.reshape(a_size[0])
    #
    # # a = a.reshape(25, 81, 82, 128)
    # print(a.size())

    """测试1x1卷积"""
    # from schnetpack.nn.base import Dense
    # a = torch.randn(25, 85, 86, 25)
    # model = Dense(25, 128)
    # y = model(a)
    # print(y.size())

    """测试relu"""
    #
    # from caSchnetpack.activationFunction.dy_relu import DyReLUB
    #
    # a = torch.randn(25, 85, 86, 25)
    # model = DyReLUB()
    # y = model(a)
    # print(y)

    """测试pooling"""
    a = torch.randn(25, 85, 86, 25)
    model = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    y = model(a)
    print(y)
    print(y.size())



    """
    目前的几个想法：
    1. 更换conv1d为dense, 去掉maxpoolin 
    2. 把维度进行拼接
    """