# -*- coding: utf-8 -*-
# @Time    : 2022/3/18 22:18
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from loguru import logger
from caSchnetpack.activationFunction.dy_relu import DyReLUA
#from schnetpack.nn.base import Dense
from caSchnetpack.base.Dense import Dense
from caSchnetpack.activationFunction.acf_gelus import gelus_gt2, gelu_bert, soft_plus

def convDense_base(in_channels, out_channels,  module_name, postfix, n_gaussians=25,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    """postfix == 1, add maxpooling(3,1,1)"""
    if postfix == 0:
        return [
            ('{}_{}/conv'.format(module_name, postfix),
                Dense(n_gaussians, out_channels, bias=True)
             ),
            ('{}_{}/gelu'.format(module_name, postfix),
                gelus_gt2()),
        ]
    elif postfix == 2:
        return [

            ('{}_{}/conv'.format(module_name, postfix),
             Dense(in_channels, out_channels, bias=True)
             ),
            ('{}_{}/gelu'.format(module_name, postfix),
             gelus_gt2()),
            ('{}_{}/maxpooling'.format(module_name, postfix),
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        ]
    else:
        return [
            ('{}_{}/conv'.format(module_name, postfix),
             Dense(in_channels, out_channels, bias=True)
             ),
            ('{}_{}/dy_relu'.format(module_name, postfix),
             gelus_gt2()),
        ]



def convDense_contact(in_channels, out_channels, module_name, postfix,
            ):

    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         Dense(in_channels, out_channels, )
         ),
    ]

class CFCA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 n_gaussians,
                 layer_per_block,
                 module_name,
                 identity=False):
        super(CFCA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(convDense_base(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = n_gaussians + layer_per_block * stage_ch
        logger.info(in_channel)

        self.concat = nn.Sequential(
            OrderedDict(convDense_contact(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            # print("x", x.size())
            output.append(x)
        # print("output", len(output))
        # print(output)
        x = torch.cat(output, dim=3)
        # print("cat",x.size())
        xt = self.concat(x)
        if self.identity:
            xt = xt + identity_feat
        # xt = xt.permute(0, 2, 1)
        return xt




# if __name__ == "__main__":
#
#     module_name = 'DGA_layer'
#     model = CFCA_module(in_ch=128, stage_ch=128, concat_ch=128, n_gaussians=25, layer_per_block=4, module_name=module_name)
#     print(model)
#     a = torch.randn(25, 85, 86, 25)
#     print(a.size())
#     # a = a.permute(0, 2, 1)
#     y = model(a)
#     print(y)
#     print(y.size())
#     # y = y.permute(0, 2, 1)







