#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/1/12 14:17
# @Author: yzf
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder
from models.exp_arch.arch_miccai.unet_edge import EGModule_AddConv

class UNetEdgeEF(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=True, conv_layer_order='cbr', init_channel_number=16):
        super(UNetEdgeEF, self).__init__()

        self.no_class = out_channels

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
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
        dec3 = self.decoders[0](enc3, mid_edge)

        dec3_edge = self.ef2(dec3, edge_feat)
        dec2 = self.decoders[1](enc2, dec3_edge)

        dec2_edge = self.ef3(dec2, edge_feat)
        dec1 = self.decoders[2](enc1, dec2_edge)

        seg_score = self.final_conv(dec1)

        return seg_score, edge_score

class EdgeFusion(nn.Module):
    def __init__(self, feat_chan, edge_chan):
        super(EdgeFusion, self).__init__()
        self.trans = nn.Sequential(nn.Conv3d(feat_chan + edge_chan, feat_chan, kernel_size=1, stride=1),
                                   nn.BatchNorm3d(feat_chan),
                                   nn.ReLU())

    def forward(self, in_feat, edge_feat):
        down_edge = F.interpolate(edge_feat, in_feat.size()[2:], mode='trilinear', align_corners=True)
        return self.trans(torch.cat((in_feat, down_edge), dim=1))

if __name__ == '__main__':
 model = UNetEdgeEF(1, 9, init_channel_number=16, conv_layer_order='cbr', interpolate=True)
 os.environ['CUDA_VISIBLE_DEVICES'] = '1'
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model = model.to(device)
 summary(model, (1, 160, 160, 64), batch_size=2)
 # torch.save(model, 'unet3d.pth')