#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/25/2021 2:32 PM
# @Author: yzf
"""Add multi-level guidance"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv

class UNetEdgeMLGdn_FnlNoEdge_DblSpv(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, conv_layer_order='cbr', init_ch=16):
        super(UNetEdgeMLGdn_FnlNoEdge_DblSpv, self).__init__()

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

        # # for dec1
        # self.fnl_trans = nn.Sequential(
        #     nn.Conv3d(init_ch * 2, init_ch, 3, 1, 1),
        #     nn.BatchNorm3d(init_ch),
        #     nn.ReLU(),
        # )
        #
        # self.fnl_enc = nn.Sequential(
        #     nn.Conv3d(init_ch, init_ch, 3, 1, 1),
        #     nn.BatchNorm3d(init_ch),
        #     nn.ReLU(),
        #     nn.Conv3d(init_ch, init_ch, 3, 1, 1),
        #     nn.BatchNorm3d(init_ch),
        #     nn.ReLU(),
        # )

        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(init_ch, self.no_class, 1))

        # for dec2
        self.sec_enc = nn.Sequential(
            nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
            nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
        )

        self.sec_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(init_ch * 2, self.no_class, 1))

        # for dec3
        self.trd_trans = nn.Sequential(
            nn.Conv3d(init_ch * 4, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
        )

        self.trd_enc = nn.Sequential(
            nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
            nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
        )

        self.trd_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                      nn.Conv3d(init_ch * 2, self.no_class, 1))

    def forward(self, x):
        # encoder part
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        mid = self.encoders[3](enc3)
        encoders_features = [enc3, enc2, enc1]

        edge_feat, edge_score = self.edge_module(enc2, mid)

        dec3 = self.decoders[0](enc3, mid)
        dec2 = self.decoders[1](enc2, dec3)
        dec1 = self.decoders[2](enc1, dec2)

        # The edge feature is diluted, so it's quite reasonable that there is no performance gain.
        # final guidance
        # diluted_ed_feat = self.fnl_trans(edge_feat)
        # diluted_ed_feat = F.interpolate(diluted_ed_feat, scale_factor=2, mode='trilinear')
        # fnl_feat = dec1 + diluted_ed_feat
        # fnl_feat = self.fnl_enc(fnl_feat)

        fnl_score = self.final_conv(dec1)

        # second guidance
        sec_feat = edge_feat + dec2
        sec_feat = self.sec_enc(sec_feat)

        sec_score = self.sec_conv(sec_feat)
        sec_score = F.interpolate(sec_score, scale_factor=2, mode='trilinear')

        # third guidance
        trd_feat = self.trd_trans(dec3)
        trd_feat = F.interpolate(trd_feat, scale_factor=2, mode='trilinear')
        trd_feat = edge_feat + trd_feat
        trd_feat = self.trd_enc(trd_feat)

        trd_score = self.trd_conv(trd_feat)
        trd_score = F.interpolate(trd_score, scale_factor=2, mode='trilinear')

        # combine all score
        seg_score = fnl_score + sec_score + trd_score
        return fnl_score, sec_score, trd_score, seg_score, edge_score

class EGModule(nn.Module):
    def __init__(self, init_ch):
        super(EGModule, self).__init__()
        self.h_conv = nn.Sequential(
            nn.Conv3d(init_ch * 8, init_ch * 2, 3, 1, 1),
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

        feat = h_feat + l_feat
        edge_feat = self.e_conv(feat)

        edge_score = self.out_conv(edge_feat)
        edge_score = F.interpolate(edge_score, scale_factor=2, mode='trilinear')
        return edge_feat, edge_score

if __name__ == '__main__':
 model = UNetEdgeMLGdn_FnlNoEdge_DblSpv(1, 9, init_ch=16, conv_layer_order='cbr', interpolate=True)
 os.environ['CUDA_VISIBLE_DEVICES'] = '2'
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model = model.to(device)
 summary(model, (1, 160, 160, 64), batch_size=4)