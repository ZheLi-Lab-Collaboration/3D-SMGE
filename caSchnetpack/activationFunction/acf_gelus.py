# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 19:35
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class gelus_gt2(nn.Module):
    def __init__(self):
        super(gelus_gt2, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class gelu_bert(nn.Module):
    def __init__(self):
        super(gelu_bert, self).__init__()

    def forward(self, x):

        cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        return cdf * x


class soft_plus(nn.Module):
    def __init__(self):
        super(soft_plus, self).__init__()

    def forward(self, x):
        return F.softplus(x) - np.log(2.0)



def gelus_gt2_fun(x):

    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelus_bert_fun(x):
    cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    return cdf * x


def soft_plus_fun(x):
    return F.softplus(x) - np.log(2.0)



