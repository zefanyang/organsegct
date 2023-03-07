#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/25/2021 2:32 PM
# @Author: yzf
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv

class EdgeDetector(nn.Module):
    def __init__(self, init_ch):
        super(EdgeDetector, self).__init__()
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

        feat = F.relu(h_feat + l_feat)
        edge_feat = self.e_conv(feat)

        edge_score = self.out_conv(edge_feat)
        edge_score = F.interpolate(edge_score, scale_factor=2, mode='trilinear')
        return edge_feat, edge_score

class EdgeAtt(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear')
        self.trans = nn.Sequential(nn.Conv3d(in_ch, 16, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm3d(16),
                                   nn.ReLU())

        # self.deep_sup = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),)
        self.deep_sup = nn.Sequential(nn.Conv3d(16, out_ch, kernel_size=1, stride=1, padding=0), )

        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(16)

    def forward(self, x, edge_score):
        x = self.trans(x)
        x = self.upsample(x)
        shortcut = x
        gated_x = x * edge_score

        # When edge_score is detached, using addition and relu
        # produces segmentations that have holes and look like contours.
        # However, we want consistency on edge and segmentation.
        # out = F.relu(shortcut + gated_x)

        out = F.relu(self.bn1(gated_x) + self.bn2(shortcut))

        # out = torch.cat((shortcut, gated_x), dim=1)

        out = self.deep_sup(out)
        return out

class UNetEdgeAtt(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, conv_layer_order='cbr', init_ch=16):
        super(UNetEdgeAtt, self).__init__()

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

        self.edge_module = EdgeDetector(init_ch)

        self.edge_att3 = EdgeAtt(in_ch=4*init_ch, out_ch=self.no_class, scale_factor=4)
        self.edge_att2 = EdgeAtt(in_ch=2*init_ch, out_ch=self.no_class, scale_factor=2)
        self.edge_att1 = EdgeAtt(in_ch=1*init_ch, out_ch=self.no_class, scale_factor=1)

        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(self.no_class * 3, self.no_class, 1))

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

        # not detach from graph
        # att_edge = edge_score.detach()
        att_edge = edge_score

        dsup3 = self.edge_att3(dec3, att_edge)
        dsup2 = self.edge_att2(dec2, att_edge)
        dsup1 = self.edge_att1(dec1, att_edge)

        # final guidance
        seg_score = self.final_conv(torch.cat((dsup3, dsup2, dsup1), dim=1))
        return seg_score, edge_score



if __name__ == '__main__':
 model = UNetEdgeAtt(1, 9, init_ch=16, conv_layer_order='cbr', interpolate=True)
 os.environ['CUDA_VISIBLE_DEVICES'] = '1'
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model = model.to(device)
 summary(model, (1, 160, 160, 64))