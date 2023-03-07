#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/3/2021 8:42 PM
# @Author: yzf
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv
from models.utils_graphical_model import DAG_RNN_4Neigh

class UNetRFP_DAGs(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, conv_layer_order='cbr', init_ch=16):
        super(UNetRFP_DAGs, self).__init__()

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

        self.rfp = RFP_DAGs(in_ch=8 * init_ch)

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

        ehn_mid = self.rfp(mid)

        dec3 = self.decoders[0](enc3, mid)
        dec2 = self.decoders[1](enc2, dec3)
        dec1 = self.decoders[2](enc1, dec2)

        seg_score = self.final_conv(dec1)

        return seg_score

class RFP_DAGs(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.dag_list = nn.ModuleList([DAG_RNN_4Neigh(in_ch) for _ in range(64//8)])  # hard-coding '64//8'

    def forward(self, x):
        hidden_x = []
        d = x.shape[-1]
        for i in range(d):
            hid = self.dag_list[i](x[..., i])
            hidden_x.append(hid.unsqueeze(-1))
        hidden_x = torch.cat(hidden_x, dim=-1)

        return F.relu(x + hidden_x)

if __name__ == '__main__':
    import os
    from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = UNetRFP_DAGs(in_ch=1, out_ch=9).cuda()
    summary(model, (1, 160, 160, 64))
