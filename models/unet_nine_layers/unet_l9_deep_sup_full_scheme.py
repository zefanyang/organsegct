#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/2/2021 9:49 AM
# @Author: yzf
"""UNet with edge skip connection, RFP-Head, and deep supervision"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv
from models.unet_nine_layers.unet_l9_deep_sup import DeepSup
from models.unet_nine_layers.unet_l9_deep_sup_edge import EGModule
from models.unet_nine_layers.unet_l9_deep_sup_edge_skip import edge_fusion
# from models.unet_nine_layers.unet_l9_deep_sup_rfp import RFP_UAGs
from models.utils_graphical_model import UAG_RNN_4Neigh, UAG_RNN_8Neigh

class RFP_UAGs(nn.Module):
    def __init__(self, in_ch, num_neigh='four'):
        super().__init__()
        self.dag_list = None
        if num_neigh == 'four':
            self.dag_list = nn.ModuleList([UAG_RNN_4Neigh(in_ch) for _ in range(64//16)])  # hard-coding '64//8'
        elif num_neigh == 'eight':
            self.dag_list = nn.ModuleList([UAG_RNN_8Neigh(in_ch) for _ in range(64//16)])  # hard-coding '64//8'
        # # add an adaption layer, which may increase learning flexibility
        # self.adapt = nn.Sequential(
        #     nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(in_ch),
        #     nn.ReLU()
        # )

    def forward(self, x):
        d = x.shape[-1]
        x_hid = []
        # x_adp = self.adapt(x)
        x_adp = x

        for i in range(d):
            hid = self.dag_list[i](x_adp[..., i])
            x_hid.append(hid.unsqueeze(-1))
        x_hid = torch.cat(x_hid, dim=-1)

        return x_adp + x_hid

class UNetL9DeepSupFullScheme(nn.Module):
    def __init__(self, in_ch, out_ch, num_neigh='four', interpolate=True, init_ch=16, conv_layer_order='cbr'):
        super(UNetL9DeepSupFullScheme, self).__init__()

        self.no_class = out_ch

        ## Encoder
        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(2 * init_ch, 4 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(4 * init_ch, 8 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(8 * init_ch, 16 * init_ch, conv_layer_order=conv_layer_order)
        ])

        ## Decoder
        self.decoders = nn.ModuleList([
            Decoder(8*init_ch+16*init_ch+32, 8*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(4*init_ch+8*init_ch+32, 4*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(2*init_ch+4*init_ch+32, 2*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(init_ch+2*init_ch+32, init_ch, interpolate, conv_layer_order=conv_layer_order)
        ])

        # deep supervision
        self.deep_sup4 = DeepSup(8 * init_ch, out_ch=self.no_class, scale_factor=8)
        self.deep_sup3 = DeepSup(4 * init_ch, out_ch=self.no_class, scale_factor=4)
        self.deep_sup2 = DeepSup(2 * init_ch, out_ch=self.no_class, scale_factor=2)
        self.deep_sup1 = nn.Conv3d(init_ch, self.no_class, kernel_size=1)

        ## Edge detection
        self.edge_module = EGModule(init_ch)

        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(self.no_class * 4, self.no_class, 1))

        ## RFP-Head
        trans_ch = 16 * init_ch // 2

        self.adapt = nn.Sequential(
            nn.Conv3d(16*init_ch, trans_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(trans_ch),
            nn.ReLU(),
        )
        self.rfp = RFP_UAGs(in_ch=trans_ch, num_neigh=num_neigh)
        self.rfp_fnl_conv = nn.Sequential(
            nn.Conv3d(trans_ch, trans_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(trans_ch),
            nn.ReLU(),
            nn.Conv3d(trans_ch, self.no_class, 1)
        )

        # Out conv
        self.comb_fnl_conv = nn.Conv3d(self.no_class * 2, self.no_class, 1)

    def forward(self, x):
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        enc4 = self.encoders[3](enc3)
        mid = self.encoders[4](enc4)
        encoders_features = [enc4, enc3, enc2, enc1]

        # Edge detection
        edge_feat, edge_score = self.edge_module(enc2, mid)

        # Edge skip-connections
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

        # RFP-Head
        mid_adapt = self.adapt(mid)
        ehn_mid = self.rfp(mid_adapt)
        rfp_seg_score = self.rfp_fnl_conv(ehn_mid)
        rfp_seg_score = F.upsample(rfp_seg_score, scale_factor=16, mode='trilinear', align_corners=True)

        comb_seg_score = self.comb_fnl_conv(torch.cat((seg_score, rfp_seg_score), 1))

        return seg_score, comb_seg_score, edge_score

if __name__ == '__main__':
    import time
    import os
    from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = UNetL9DeepSupFullScheme(1, 9, num_neigh='eight', init_ch=16, conv_layer_order='cbr', interpolate=True)
    device = torch.device('cuda')
    model = model.to(device)

    data = torch.randn((1, 1, 160, 160, 64)).cuda()
    tic = time.time()
    x = model(data)
    toc = time.time()
    print('Inference Time {:.4f}'.format(toc-tic))
    # four neighbor: 0.9381 s
    # eight neighbor: 2.3929 s

    # summary(model, (1, 160, 160, 64))

    # from models.unet_nine_layers.unet_l9 import count_parameters
    # print('Total number of trainable parameters: {:.2f} M'.format(count_parameters(model) / 1e6))