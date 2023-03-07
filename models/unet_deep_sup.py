#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/2/2021 9:49 AM
# @Author: yzf
"""UNet with deep supervision"""
# Four-layer UNet with deep supervision does not show accuracy improvement.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv

class DeepSup(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.dsup = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                  nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
    def forward(self, x):
        return self.dsup(x)

class UNetDeepSup(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, init_ch=16, conv_layer_order='cbr'):
        super(UNetDeepSup, self).__init__()

        self.no_class = out_ch

        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(2 * init_ch, 4 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(4 * init_ch, 8 * init_ch, conv_layer_order=conv_layer_order),
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_ch + 8 * init_ch, 4 * init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(2 * init_ch + 4 * init_ch, 2 * init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(init_ch + 2 * init_ch, init_ch, interpolate, conv_layer_order=conv_layer_order)
        ])


        self.deep_sup3 = DeepSup(4 * init_ch, out_ch=self.no_class, scale_factor=4)
        self.deep_sup2 = DeepSup(2 * init_ch, out_ch=self.no_class, scale_factor=2)
        self.deep_sup1 = nn.Conv3d(init_ch, self.no_class, kernel_size=1)

        # The influence of dropout 0.1 on final results is trivial.
        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(self.no_class * 3, self.no_class, 1))

    def forward(self, x):
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        mid = self.encoders[3](enc3)
        encoders_features = [enc3, enc2, enc1]

        dec3 = self.decoders[0](enc3, mid)
        dec2 = self.decoders[1](enc2, dec3)
        dec1 = self.decoders[2](enc1, dec2)

        dsup3 = self.deep_sup3(dec3)
        dsup2 = self.deep_sup2(dec2)
        dsup1 = self.deep_sup1(dec1)

        # Introducing deep supervision in 7-layer UNet shows performance drop.
        seg_score = self.final_conv(torch.cat((dsup3, dsup2, dsup1), dim=1))

        return seg_score

if __name__ == '__main__':
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = UNetDeepSup(1, 9, init_ch=16, conv_layer_order='cbr', interpolate=True)
    device = torch.device('cuda')
    model = model.to(device)
    start = time.time()
    summary(model, (1, 160, 160, 64))
    print("take {:f} s".format(time.time() - start))