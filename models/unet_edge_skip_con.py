#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/1/12 14:17
# @Author: yzf
"""2021/06/24 full resolution edge prediction and multi-level edge fusion"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv
from models.unet_nine_layers.unet_l9_deep_sup import DeepSup
from models.unet_nine_layers.unet_l9_deep_sup_edge import EGModule

class UNetEdgeSkip(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, conv_layer_order='cbr', init_ch=16):
        super(UNetEdgeSkip, self).__init__()

        self.no_class = out_ch

        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2*init_ch, conv_layer_order=conv_layer_order),
            Encoder(2*init_ch, 4*init_ch, conv_layer_order=conv_layer_order),
            Encoder(4*init_ch, 8*init_ch, conv_layer_order=conv_layer_order),
        ])

        self.decoders = nn.ModuleList([
            Decoder(4*init_ch+8*init_ch+32, 4*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(2*init_ch+4*init_ch+32, 2*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(1*init_ch+2*init_ch+32, init_ch, interpolate, conv_layer_order=conv_layer_order)
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

        edge_feat, edge_score = self.edge_module(enc2, mid)

        skip3 = edge_fusion(enc3, edge_feat)
        skip2 = edge_fusion(enc2, edge_feat)
        skip1 = edge_fusion(enc1, edge_feat)

        dec3 = self.decoders[0](skip3, mid)
        dec2 = self.decoders[1](skip2, dec3)
        dec1 = self.decoders[2](skip1, dec2)

        seg_score = self.final_conv(dec1)

        return seg_score, edge_score

def edge_fusion(skip_feat, edge_feat):
    edge_feat = F.interpolate(edge_feat, skip_feat.size()[2:], mode='trilinear')
    return torch.cat((edge_feat, skip_feat), dim=1)

class EGModule(nn.Module):
    def __init__(self, init_ch):
        super(EGModule, self).__init__()

        # Use 3*3*3 or 1*1*1 convolution to concentrate the channel.
        # Empirically, 3*3*3 convolution produces better edge prediction.
        self.h_conv = nn.Sequential(
            nn.Conv3d(init_ch * 8, init_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU()
        )

        self.e_conv = nn.Sequential(
            nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
            nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
        )
        self.out_conv = nn.Conv3d(init_ch*2, 1, 1)

    def forward(self, l_feat, h_feat):
        h_feat = self.h_conv(h_feat)
        h_feat = F.interpolate(h_feat, scale_factor=4, mode='trilinear')

        # add ReLU after addition?  Show mild performance drop
        feat = F.relu(h_feat + l_feat)
        edge_feat = self.e_conv(feat)
        edge_score = self.out_conv(edge_feat)
        edge_score = F.interpolate(edge_score, scale_factor=2, mode='trilinear')
        return edge_feat, edge_score

if __name__ == '__main__':
 model = UNetEdgeSkip(1, 9, init_ch=16, conv_layer_order='cbr', interpolate=True)
 os.environ['CUDA_VISIBLE_DEVICES'] = '1'
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model = model.to(device)
 summary(model, (1, 160, 160, 64), batch_size=4)
 # torch.save(model, 'unet3d.pth')