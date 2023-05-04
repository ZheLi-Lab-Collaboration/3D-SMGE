# -*- coding: utf-8 -*-
# @Time    : 2022/3/20 15:11
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

import torch
from torch import nn

from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate
from loguru import logger



from caSchnetpack.activationFunction.dy_relu import DyReLUA

# __all__ = ["CACFConv"]


class CACFConv(nn.Module):
    r"""Continuous-filter convolution block used in SchNet module.
 
    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        n_filters (int): number of filter dimensions.
        n_out (int): number of output dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_filters,
        n_out,
        ca_filter_nn,
        atomBase_network,
        activation=None,
        cutoff_network=None,
        normalize_filter=False,
        axis=2,
    ):
        super(CACFConv, self).__init__()
        self.in2f = Dense(128, n_filters, bias=False, activation=None)


        self.atomBase_network = atomBase_network
        self.feature_out = Dense(n_filters, n_out, bias=True, activation=activation)


        self.ca_filter_nn = ca_filter_nn
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.
        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # (N_b, N_a, N_nbh, N_g) --> (N_bxN_a, N_nbh, N_g)
        # f_size = f_ij.size()
        # f_ij = f_ij.reshape(f_size[0]*f_size[1], f_size[2], f_size[3])

        # pass expanded interactomic distances through filter block
        W = self.ca_filter_nn(f_ij)

        # w_size = W.size()
        # W = W.reshape(f_size[0], f_size[1], f_size[2], w_size[2])


        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        # embedding -> x [25, 82, 128]
        y = self.in2f(x)

        # reshape y for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        # 保持前两个维度
        nbh = nbh.expand(-1, -1, y.size(2))

        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        # element-wise multiplication, aggregating and Dense layer
        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.feature_out(y)
        # print("cfconv return y", y)
        return y

