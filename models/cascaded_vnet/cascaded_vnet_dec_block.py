#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/1/2021 12:24 PM
# @Author: yzf
"""Cascaded V-Net: concatenate final output, and decoder blocks"""
import torch
import torch.nn as nn
from models.cascaded_vnet.vnet_kernel_size_3 import InputTransition, DownTransition, OutputTransition, UpTransition
from models.cascaded_vnet.cascaded_vnet_fnl_output import CombinationTransition

class ConvUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3d = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ELU()

    def forward(self, x):
        return self.relu(self.bn(self.conv3d(x)))

def passthrough(x):
    return x

class SndStgUpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(outChans // 2),
            nn.ELU()
        )
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.merge_conv = ConvUnit(in_ch=(outChans//2)*3 , out_ch=outChans)
        self.relu2 = nn.ELU()
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = nn.Sequential(*[ConvUnit(outChans, outChans) for _ in range(nConvs)])

    def forward(self, x, skipx, fst_stg_skip):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.up_conv(out)  # 256 -> 128
        xcat = torch.cat((out, skipxdo, fst_stg_skip), 1)  # (128 cat 128) cat 128
        xcat = self.merge_conv(xcat)  # conv -> 256
        out = self.ops(xcat)  # 256 -> 256
        out = self.relu2(torch.add(out, xcat))  # shortcut connection
        return out

class FirstStageVNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True):
        super(FirstStageVNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        # The performance decreases when dropout is closed
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)

        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out, out16, out32, out64, out128

class SecondStageVNet_DecBlock(nn.Module):
    def __init__(self, elu=True):
        super(SecondStageVNet_DecBlock, self).__init__()
        self.in_tr = CombinationTransition(1 + 9, 16)  # concatenate input and seg_score
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        # The performance decreases when dropout is closed
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)

        self.up_tr256 = SndStgUpTransition(256, 256, 2, dropout=True)
        self.up_tr128 = SndStgUpTransition(256, 128, 2, dropout=True)
        self.up_tr64 = SndStgUpTransition(128, 64, 1)
        self.up_tr32 = SndStgUpTransition(64, 32, 1)
        self.out_tr = OutputTransition(32, elu)

    def forward(self, x, seg_score, fst_stg_dec):
        out16 = self.in_tr(x, seg_score)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128, fst_stg_dec[-1])
        out = self.up_tr128(out, out64, fst_stg_dec[-2])
        out = self.up_tr64(out, out32, fst_stg_dec[-3])
        out = self.up_tr32(out, out16, fst_stg_dec[-4])
        out = self.out_tr(out)
        return out

class CascadedNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = FirstStageVNet()
        self.net2 = SecondStageVNet_DecBlock()

    def forward(self, x):
        out = self.net1(x)
        score2 = self.net2(x, seg_score=out[0], fst_stg_dec=out[1:])
        return out, score2

if __name__ == '__main__':
    import os
    from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device('cuda')
    model = CascadedNetworks()
    model = model.to(dev)

    from models.unet_nine_layers.unet_l9 import count_parameters
    print('Total number of trainable parameters: {:.2f} M'.format(count_parameters(model) / 1e6))