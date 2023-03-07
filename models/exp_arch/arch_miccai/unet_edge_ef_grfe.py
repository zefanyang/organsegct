#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 1/20/2021 8:11 PM
# @Author: yzf
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/1/12 14:17
# @Author: yzf
import os
from typing import  Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ReLU
from models.unet import Encoder, Decoder
from models.exp_arch.arch_miccai.unet_edge_ef import EGModule_AddConv, EdgeFusion

class GRFENet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_size,
        interpolate=True,
        conv_layer_order='cbr',
        init_channel_number=16,
        init_stgy: Optional[str]=None):
        super().__init__()

        self.no_class = out_channels
        self.input_size = input_size

        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order=conv_layer_order),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order=conv_layer_order),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order=conv_layer_order),
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate,
                    conv_layer_order=conv_layer_order),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
                    conv_layer_order=conv_layer_order),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
                    conv_layer_order=conv_layer_order)
        ])

        self.edge_module = EGModule_AddConv(init_channel_number)
        self.ef1 = EdgeFusion(8*init_channel_number, 2*init_channel_number)
        self.ef2 = EdgeFusion(4*init_channel_number, 2*init_channel_number)
        self.ef3 = EdgeFusion(2*init_channel_number, 2*init_channel_number)

        self.bip_head = GBIPHead(8*init_channel_number, self.input_size[2]//8, out_channels=self.no_class)
        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(init_channel_number, self.no_class, 1))

    def forward(self, x):
        # encoder part
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        mid = self.encoders[3](enc3)
        edge_feat, edge_score = self.edge_module(enc2, mid, x.size())
        mid_edge = self.ef1(mid, edge_feat)

        down_size = [s//8 for s in self.input_size]
        down_edge_score = F.upsample(edge_score, size=down_size, mode='trilinear', align_corners=True)  # down_size
        feat_sum = self.bip_head(mid_edge, down_edge_score)

        dec3 = self.decoders[0](enc3, feat_sum)

        dec3_edge = self.ef2(dec3, edge_feat)
        dec2 = self.decoders[1](enc2, dec3_edge)

        dec2_edge = self.ef3(dec2, edge_feat)
        dec1 = self.decoders[2](enc1, dec2_edge)

        seg_score = self.final_conv(dec1)

        return seg_score, edge_score

class GBIPHead(nn.Module):
    """Graph-based boundary-aware information propagation"""
    def __init__(self, in_channels, no_slice, out_channels):
        super(GBIPHead, self).__init__()
        # self.mode = mode
        self.no_slice = no_slice
        inter_channels = in_channels // 1

        self.adapt1 = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm3d(inter_channels),
                                    nn.ReLU())
        self.adapt2 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, dilation=12, padding=12, bias=False),
                                    nn.BatchNorm3d(inter_channels),
                                    nn.ReLU())
        self.adapt3 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, dilation=12, padding=12, bias=False),
                                    nn.BatchNorm3d(inter_channels),
                                    nn.ReLU())

        # 3d UAG
        self.uag3d_rnn = nn.ModuleList()
        for i in range(self.no_slice):
            self.uag3d_rnn.append(UAG_RNN_4Neigh(inter_channels))

        # learnable parameters
        self.gamma = nn.Parameter(2*torch.ones(1))  # tensor([2.])
        self.bias = nn.Parameter(torch.ones(1)/out_channels)  # tensor([1./no_class])

    def forward(self, x, boundary_):
        """
        :param x:
        :param boundary_: edge score
        :return:
        """
        # adapt from CNN
        feat1 = self.adapt1(x)
        feat2 = self.adapt2(feat1)

        # boundary confidence
        # method 1
        boundary = 1 - torch.sigmoid(10.*boundary_-2.)*self.gamma
        # method 2
        # boundary = torch.mean(torch.mean(boundary_, 2, True), 3, True)-boundary_+self.bias
        # boundary = (boundary - torch.min(torch.min(boundary, 3, True)[0], 2, True)[0])*self.gamma

        boundary = torch.clamp(boundary, max=1)
        boundary = torch.clamp(boundary, min=0)

        # UAG
        tmp = []
        for i in range(self.no_slice):
            x_2d = feat2[:, :, :, :, i]
            y_2d = boundary[:, :, :, :, i]
            tmp.append(self.uag3d_rnn[i](x_2d, y_2d)[...,None])
        graph = torch.cat(tmp, dim=-1)

        # if self.mode == '3d_context':
        #     graph = self.adapt3(graph)

        feat_sum = graph + feat2  # residual connection to boost gradient flow

        return feat_sum

class UAG_RNN_4Neigh(nn.Module):
    """Unidirectional Acyclic Graphs (UCGs)"""

    def __init__(self, in_dim):
        super(UAG_RNN_4Neigh, self).__init__()
        self.chanel_in = in_dim
        self.relu = ReLU()

        # self.gamma1 = Parameter(0.5 * torch.ones(1))
        # self.gamma2 = Parameter(0.5 * torch.ones(1))
        # self.gamma3 = Parameter(0.5 * torch.ones(1))
        # self.gamma4 = Parameter(0.5 * torch.ones(1))
        self.conv1 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv2 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # south
        # self.conv3 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv4 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv5 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # east
        # self.conv6 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv7 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv8 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # west

        self.conv9 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv10 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # north
        # self.conv11 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv12 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv13 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # east
        # self.conv14 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv15 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv16 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # west

    def forward(self, x, y):
        m_batchsize, C, height, width = x.size()

        ## s plane
        hs = x * 1
        for i in range(height):
            if i > 0:
                hs[:, :, i, :] = self.conv1(hs[:, :, i, :].clone()) + self.conv2(hs[:, :, i - 1, :].clone()) * y[:, :,i - 1, :]
                hs[:, :, i, :] = self.relu(hs[:, :, i, :].clone())

        ## e plane
        hse = hs * 1
        for j in range(width):
            if j > 0:
                hse[:, :, :, j] = self.conv4(hse[:, :, :, j].clone()) + self.conv5(hse[:, :, :, j - 1].clone()) * y[:,:, :,j - 1]
            hse[:, :, :, j] = self.relu(hse[:, :, :, j].clone())

        ## w plane
        hsw = hs * 1
        for j in reversed(range(width)):
            if j < (width - 1):
                hsw[:, :, :, j] = self.conv7(hsw[:, :, :, j].clone()) + self.conv8(hsw[:, :, :, j + 1].clone()) * y[:,:, :,j + 1]
            hsw[:, :, :, j] = self.relu(hsw[:, :, :, j].clone())

        ## n plane
        hn = x * 1
        for i in reversed(range(height)):
            if i < (height - 1):
                hn[:, :, i, :] = self.conv9(hn[:, :, i, :].clone()) + self.conv10(hn[:, :, i + 1, :].clone()) * y[:, :,i + 1,:]
            hn[:, :, i, :] = self.relu(hn[:, :, i, :].clone())

        ## ne plane
        hne = hn * 1
        for j in range(width):
            if j > 0:
                hne[:, :, :, j] = self.conv12(hne[:, :, :, j].clone()) + self.conv13(hne[:, :, :, j - 1].clone()) * y[:,:,:,j - 1]
            hne[:, :, :, j] = self.relu(hne[:, :, :, j].clone())

        ## nw plane
        hnw = hn * 1
        for j in reversed(range(width)):
            if j < (width - 1):
                hnw[:, :, :, j] = self.conv15(hnw[:, :, :, j].clone()) + self.conv16(hnw[:, :, :, j + 1].clone()) * y[:,:,:,j + 1]
            hnw[:, :, :, j] = self.relu(hnw[:, :, :, j].clone())

        out = hse + hsw + hnw + hne
        return out

# class UNet3DEdgeRNNDSV(nn.Module):
#     def __init__(self, in_channels, out_channels, input_size, mode=None, interpolate=True, conv_layer_order='cbr', init_channel_number=16):
#         super(UNet3DEdgeRNNDSV, self).__init__()
#
#         self.no_class = out_channels
#         self.input_size = input_size
#
#         self.encoders = nn.ModuleList([
#             Encoder(in_channels, init_channel_number, is_max_pool=False, conv_layer_order=conv_layer_order),
#             Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order=conv_layer_order),
#             Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order=conv_layer_order),
#             Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order=conv_layer_order),
#         ])
#
#         self.decoders = nn.ModuleList([
#             Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, interpolate,
#                     conv_layer_order=conv_layer_order),
#             Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, interpolate,
#                     conv_layer_order=conv_layer_order),
#             Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, interpolate,
#                     conv_layer_order=conv_layer_order)
#         ])
#
#         self.edge_module = EGModule_AddConv(init_channel_number)
#         self.bip_head = GBIPHead(8*init_channel_number, self.input_size[2]//8, mode)
#
#         self.dsv3 = UNetDSV(4 * init_channel_number, out_size=self.no_class, scale_factor=4)
#         self.dsv2 = UNetDSV(2 * init_channel_number, out_size=self.no_class, scale_factor=2)
#         self.dsv1 = nn.Conv3d(init_channel_number, self.no_class, 1)
#
#         self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
#                                         nn.Conv3d(3*self.no_class, self.no_class, 1))
#         # self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)
#
#     def forward(self, x):
#         # encoder part
#         encoders_features = []
#         enc1 = self.encoders[0](x)
#         enc2 = self.encoders[1](enc1)
#         enc3 = self.encoders[2](enc2)
#         mid = self.encoders[3](enc3)
#         encoders_features = [enc3, enc2, enc1]
#
#         edge_score = self.edge_module(enc2, mid, x.size())
#         down_size = [s//8 for s in self.input_size]
#         down_edge_score1 = F.upsample(edge_score, size=down_size, mode='trilinear', align_corners=True)  # down_size
#
#         feat_sum1 = self.bip_head(mid, down_edge_score1)
#         dec3 = self.decoders[0](enc3, feat_sum1)
#         dec2 = self.decoders[1](enc2, dec3)
#         dec1 = self.decoders[2](enc1, dec2)
#
#         dsv3 = self.dsv3(dec3)
#         dsv2 = self.dsv2(dec2)
#         dsv1 = self.dsv1(dec1)
#
#         seg_score = self.final_conv(torch.cat([dsv1, dsv2, dsv3], dim=1))
#         return seg_score, edge_score

# class UNetDSV(nn.Module):
#     def __init__(self, in_size, out_size, scale_factor):
#         super(UNetDSV, self).__init__()
#         self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
#                                  nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )
#
#     def forward(self, input):
#         return self.dsv(input)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRFENet(1, 9, input_size=(160, 160, 64), init_channel_number=16, conv_layer_order='cbr', interpolate=True, init_stgy='he')
    # model = model.to(device)
    # start = time.time()
    # summary(model, (1, 160, 160, 64))
    # print("take {:f} s".format(time.time() - start))
