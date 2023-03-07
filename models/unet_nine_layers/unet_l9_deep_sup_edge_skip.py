#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/2/2021 9:49 AM
# @Author: yzf
"""UNet edge skip connection with deep supervision"""
# Four-layer UNet with deep supervision does not show accuracy improvement.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv
from models.unet_nine_layers.unet_l9_deep_sup import DeepSup
from models.unet_nine_layers.unet_l9_deep_sup_edge import EGModule

def edge_fusion(skip_feat, edge_feat):
    edge_feat = F.interpolate(edge_feat, skip_feat.size()[2:], mode='trilinear')
    return torch.cat((edge_feat, skip_feat), dim=1)

class UNetL9DeepSupEdgeSkip(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, init_ch=16, conv_layer_order='cbr'):
        super(UNetL9DeepSupEdgeSkip, self).__init__()

        self.no_class = out_ch

        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(2 * init_ch, 4 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(4 * init_ch, 8 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(8 * init_ch, 16 * init_ch, conv_layer_order=conv_layer_order)
        ])

        self.decoders = nn.ModuleList([
            Decoder(8*init_ch+16*init_ch+32, 8*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(4*init_ch+8*init_ch+32, 4*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(2*init_ch+4*init_ch+32, 2*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(init_ch+2*init_ch+32, init_ch, interpolate, conv_layer_order=conv_layer_order)
        ])

        self.deep_sup4 = DeepSup(8 * init_ch, out_ch=self.no_class, scale_factor=8)
        self.deep_sup3 = DeepSup(4 * init_ch, out_ch=self.no_class, scale_factor=4)
        self.deep_sup2 = DeepSup(2 * init_ch, out_ch=self.no_class, scale_factor=2)
        self.deep_sup1 = nn.Conv3d(init_ch, self.no_class, kernel_size=1)

        self.edge_module = EGModule(init_ch)

        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(self.no_class * 4, self.no_class, 1))

    def forward(self, x):
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        enc4 = self.encoders[3](enc3)
        mid = self.encoders[4](enc4)
        encoders_features = [enc4, enc3, enc2, enc1]

        edge_feat, edge_score = self.edge_module(enc2, mid)

        skip4 = edge_fusion(enc4, edge_feat)
        skip3 = edge_fusion(enc3, edge_feat)
        skip2 = edge_fusion(enc2, edge_feat)
        skip1 = edge_fusion(enc1, edge_feat)

        dec4 = self.decoders[0](skip4, mid)
        dec3 = self.decoders[1](skip3, dec4)
        dec2 = self.decoders[2](skip2, dec3)
        dec1 = self.decoders[3](skip1, dec2)

        dsup4 = self.deep_sup4(dec4)
        dsup3 = self.deep_sup3(dec3)
        dsup2 = self.deep_sup2(dec2)
        dsup1 = self.deep_sup1(dec1)

        seg_score = self.final_conv(torch.cat((dsup4, dsup3, dsup2, dsup1), dim=1))

        return seg_score, edge_score

if __name__ == '__main__':
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = UNetL9DeepSupEdgeSkip(1, 9, init_ch=16, conv_layer_order='cbr', interpolate=True)
    device = torch.device('cuda')
    model = model.to(device)

    from models.unet_nine_layers.unet_l9 import count_parameters
    print('Total number of trainable parameters: {:.2f} M'.format(count_parameters(model) / 1e6))