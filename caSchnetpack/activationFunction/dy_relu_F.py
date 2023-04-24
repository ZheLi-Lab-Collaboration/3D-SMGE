# -*- coding: utf-8 -*-
# @Time    : 2022/3/20 22:09
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

from caSchnetpack.activationFunction.dy_relu import DyReLUA



def dy_relu_F(x, channels=128, conv_type="1d"):
    dy_relu = DyReLUA(channels, conv_type=conv_type)
    return dy_relu(x)
