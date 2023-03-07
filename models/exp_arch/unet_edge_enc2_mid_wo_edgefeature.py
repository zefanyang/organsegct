#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/28/2021 9:58 PM
# @Author: yzf
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv

class UNetEdge_NoEdgeFeature(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, conv_layer_order='cbr', init_ch=16):
        super(UNetEdge_NoEdgeFeature, self).__init__()

        self.no_class = out_ch

        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(2 * init_ch, 4 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(4 * init_ch, 8 * init_ch, conv_layer_order=conv_layer_order),
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_ch + 8 * init_ch, 4 * init_ch, interpolate,
                    conv_layer_order=conv_layer_order),
            Decoder(2 * init_ch + 4 * init_ch, 2 * init_ch, interpolate,
                    conv_layer_order=conv_layer_order),
            Decoder(init_ch + 2 * init_ch, init_ch, interpolate,
                    conv_layer_order=conv_layer_order)
        ])

        self.edge_module = EGModule(init_ch)
        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(init_ch, self.no_class, 1))

    def forward(self, x):
        # encoder part
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        mid = self.encoders[3](enc3)
        encoders_features = [enc3, enc2, enc1]

        _, edge_score = self.edge_module(enc2, mid)

        dec3 = self.decoders[0](enc3, mid)
        dec2 = self.decoders[1](enc2, dec3)
        dec1 = self.decoders[2](enc1, dec2)

        seg_score = self.final_conv(dec1)

        return seg_score, edge_score

class EGModule(nn.Module):
    def __init__(self, init_ch):
        super(EGModule, self).__init__()

        # 1*1*1 convolution to concentrate the channel
        self.h_conv = nn.Sequential(
            nn.Conv3d(init_ch * 8, init_ch * 2, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU()
        )
        # self.e_conv = nn.Sequential(
        #     nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
        #     nn.BatchNorm3d(init_ch * 2),
        #     nn.ReLU(),
        #     nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
        #     nn.BatchNorm3d(init_ch * 2),
        #     nn.ReLU(),
        # )
        self.out_conv = nn.Conv3d(init_ch*2, 1, 1)

    def forward(self, l_feat, h_feat):
        h_feat = self.h_conv(h_feat)
        h_feat = F.interpolate(h_feat, scale_factor=4, mode='trilinear')

        feat = h_feat + l_feat
        edge_feat = feat

        edge_score = self.out_conv(edge_feat)
        edge_score = F.interpolate(edge_score, scale_factor=2, mode='trilinear')
        return edge_feat, edge_score

if __name__ == '__main__':
 model = UNetEdge_NoEdgeFeature(1, 9, init_ch=16, conv_layer_order='cbr', interpolate=True)
 os.environ['CUDA_VISIBLE_DEVICES'] = '0'
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model = model.to(device)
 summary(model, (1, 160, 160, 64))
 # torch.save(model, 'unet3d.pth')