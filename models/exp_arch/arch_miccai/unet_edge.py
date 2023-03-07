#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/1/12 14:17
# @Author: yzf
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.unet import Encoder, Decoder, DoubleConv

class UNetEdge(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=True, conv_layer_order='cbr', init_channel_number=16):
        super(UNetEdge, self).__init__()

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
        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(init_channel_number, self.no_class, 1))

    def forward(self, x):
        # encoder part
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        mid = self.encoders[3](enc3)
        encoders_features = [enc3, enc2, enc1]

        _, edge_score = self.edge_module(enc2, mid, x.size())

        dec3 = self.decoders[0](enc3, mid)
        dec2 = self.decoders[1](enc2, dec3)
        dec1 = self.decoders[2](enc1, dec2)

        seg_score = self.final_conv(dec1)

        return seg_score, edge_score

class EGModule_AddConv(nn.Module):
    """add and conv"""
    # ReLU in final layer
    def __init__(self, init_channels):
        super(EGModule_AddConv, self).__init__()
        self.up_num = 2
        self.feat = nn.Sequential(nn.Conv3d(init_channels*2, init_channels*2, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(init_channels*2),
                                  nn.ReLU(),
                                  nn.Conv3d(init_channels*2, init_channels*2, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(init_channels*2),
                                  nn.ReLU(),)  # TODO

        self.up_and_conv = self.high_stage_up(init_channels*8)
        self.score = nn.Sequential(nn.Dropout3d(0.1, False),
                                   nn.Conv3d(init_channels*2, 1, kernel_size=3, padding=1))

    def high_stage_up(self, in_channels):
        ops = []
        this_channels = in_channels
        for i in range(self.up_num):
            ops.append(nn.ConvTranspose3d(this_channels, this_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            ops.append(nn.Sequential(nn.Conv3d(this_channels, this_channels//2, kernel_size=3, padding=1),
                                     nn.BatchNorm3d(this_channels//2),
                                     nn.ReLU(),
                                     nn.Conv3d(this_channels//2, this_channels//2, kernel_size=3, padding=1),
                                     nn.BatchNorm3d(this_channels//2),
                                     nn.ReLU(),))
            this_channels = this_channels//2
        return nn.Sequential(*ops)

    def forward(self, low_stage, high_stage, imsize):
        # up and conv
        high_feat = self.up_and_conv(high_stage)
        # merge and conv
        merge = low_stage + high_feat
        edge_feat = self.feat(merge)
        edge_score = F.interpolate(self.score(edge_feat), imsize[2:], mode='trilinear', align_corners=True)

        # # boundary confidence to propagation confidence, method 1
        # prop_confidence = 1 - self.sigmoid(20*edge_score-4)*self.gamma
        # prop_confidence = torch.clip(prop_confidence, 0., 1.)

        return edge_feat, edge_score

# class EGModule_Att(nn.Module):
#     def __init__(self, init_channels, num_groups):
#         super(EGModule_Att, self).__init__()
#         self.up_num = 3
#         self.num_groups = num_groups
#         self.feat = nn.Sequential(nn.Conv3d(init_channels*2, init_channels*2, kernel_size=3, padding=1),
#                                   nn.ReLU(),
#                                   nn.GroupNorm(num_groups=self.num_groups, num_channels=init_channels*2),
#                                   nn.Conv3d(init_channels*2, init_channels*2, kernel_size=3, padding=1),
#                                   nn.ReLU(),
#                                   nn.GroupNorm(num_groups=self.num_groups, num_channels=init_channels*2),)
#         self.up_and_conv = self.non_local_block(init_channels*16)
#         self.out_conv = nn.Conv3d(init_channels*2, 1, kernel_size=3, padding=1)
#
#     def non_local_block(self, in_channels):
#         ops = []
#         this_channels = in_channels
#         for i in range(self.up_num):
#             ops.append(nn.ConvTranspose3d(this_channels, this_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
#             ops.append(nn.Sequential(nn.Conv3d(this_channels, this_channels//2, kernel_size=3, padding=1),
#                                      nn.ReLU(),
#                                      nn.GroupNorm(num_groups=self.num_groups, num_channels=this_channels // 2),
#                                      nn.Conv3d(this_channels//2, this_channels//2, kernel_size=3, padding=1),
#                                      nn.ReLU(),
#                                      nn.GroupNorm(num_groups=self.num_groups, num_channels=this_channels // 2),))
#             this_channels = this_channels//2
#         return nn.Sequential(*ops)
#
#     def forward(self, high_stage, low_stage):
#         features = self.feat(high_stage)
#         non_local = self.up_and_conv(low_stage)
#         merge = features + (features * non_local)
#         out = F.interpolate(self.out_conv(merge), scale_factor=2, mode='trilinear', align_corners=True)
#         return out
#
# class EGModule_Add(nn.Module):
#     """directly add"""
#     def __init__(self, init_channels, num_groups):
#         super(EGModule_Add, self).__init__()
#         self.up_num = 3
#         self.num_groups = num_groups
#         self.feat = nn.Sequential(nn.Conv3d(init_channels*2, init_channels*2, kernel_size=3, padding=1),
#                                   nn.ReLU(),
#                                   nn.GroupNorm(num_groups=self.num_groups, num_channels=init_channels*2),
#                                   nn.Conv3d(init_channels*2, init_channels*2, kernel_size=3, padding=1),
#                                   nn.ReLU(),
#                                   nn.GroupNorm(num_groups=self.num_groups, num_channels=init_channels*2),)
#         self.up_and_conv = self.non_local_block(init_channels*16)
#         self.out_conv = nn.Conv3d(init_channels*2, 1, kernel_size=3, padding=1)
#
#     def non_local_block(self, in_channels):
#         ops = []
#         this_channels = in_channels
#         for i in range(self.up_num):
#             ops.append(nn.ConvTranspose3d(this_channels, this_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
#             ops.append(nn.Sequential(nn.Conv3d(this_channels, this_channels//2, kernel_size=3, padding=1),
#                                      nn.ReLU(),
#                                      nn.GroupNorm(num_groups=self.num_groups, num_channels=this_channels // 2),
#                                      nn.Conv3d(this_channels//2, this_channels//2, kernel_size=3, padding=1),
#                                      nn.ReLU(),
#                                      nn.GroupNorm(num_groups=self.num_groups, num_channels=this_channels // 2),))
#             this_channels = this_channels//2
#         return nn.Sequential(*ops)
#
#     def forward(self, high_stage, low_stage):
#         features = self.feat(high_stage)
#         non_local = self.up_and_conv(low_stage)
#         merge = features + non_local
#         out_score = F.interpolate(self.out_conv(merge), scale_factor=2, mode='trilinear', align_corners=True)
#         return out_score

if __name__ == '__main__':
 model = UNetEdge(1, 9, init_channel_number=16, conv_layer_order='cbr', interpolate=True)
 os.environ['CUDA_VISIBLE_DEVICES'] = '1'
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model = model.to(device)
 summary(model, (1, 224, 224, 56), batch_size=4)
 # torch.save(model, 'unet3d.pth')