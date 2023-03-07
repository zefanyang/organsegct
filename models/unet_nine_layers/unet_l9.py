#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/2/2021 9:49 AM
# @Author: yzf
"""9-layer UNet"""
# Four-layer UNet with deep supervision does not show accuracy improvement.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv

def count_parameters(model):
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_size

def parameter_table(model):
    from prettytable import PrettyTable
    table = PrettyTable(['name', 'parameter number'])
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        param_size = param.numel()
        table.add_row([name, param_size])
    return table

class UNetL9(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, init_ch=16, conv_layer_order='cbr'):
        super(UNetL9, self).__init__()

        self.no_class = out_ch

        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(2 * init_ch, 4 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(4 * init_ch, 8 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(8 * init_ch, 16 * init_ch, conv_layer_order=conv_layer_order)
        ])

        self.decoders = nn.ModuleList([
            Decoder(8 * init_ch + 16 * init_ch, 8 * init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(4 * init_ch + 8 * init_ch, 4 * init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(2 * init_ch + 4 * init_ch, 2 * init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(init_ch + 2 * init_ch, init_ch, interpolate, conv_layer_order=conv_layer_order)
        ])

        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(init_ch, self.no_class, 1))

    def forward(self, x):
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        enc4 = self.encoders[3](enc3)
        mid = self.encoders[4](enc4)
        encoders_features = [enc4, enc3, enc2, enc1]

        dec4 = self.decoders[0](enc4, mid)
        dec3 = self.decoders[1](enc3, dec4)
        dec2 = self.decoders[2](enc2, dec3)
        dec1 = self.decoders[3](enc1, dec2)

        seg_score = self.final_conv(dec1)

        return seg_score

if __name__ == '__main__':
    import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')
    model = UNetL9(1, 9, init_ch=16, conv_layer_order='cbr', interpolate=True)
    model = model.to(device)
    # summary(model, (1, 160, 160, 64))
    print(model)
    print(parameter_table(model))
    print('Total number of trainable parameters: {:.2f} M'.format(count_parameters(model) / 1e6))


    # from prettytable import PrettyTable
    # table = PrettyTable(['module', 'input shape', 'output shape'])
    # x = torch.randn((1, 1, 160, 160, 64), device=device)
    # for module in model.modules():
    #     input_shape = x.shape
    #
    #     x = module(x)
    #     output_shape = x.shape
    #     table.add_row([str(module), input_shape, output_shape])
    #
    # print(table)
