#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/30/2021 11:36 AM
# @Author: yzf
"""Reproduce cascaded networks with mixed kernels in [Zhang et al. TMI 2020]"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvKnl3(nn.Module):
    """Convolutional Unit: conv + bn + relu"""
    def __init__(self, in_ch, out_ch):
        super(ConvKnl3, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvStack(nn.Module):
    """Convolutional stack: three 3*3*3 convolutional units"""
    def __init__(self, in_ch):
        super(ConvStack, self).__init__()
        self.conv_stack = nn.Sequential(
            ConvKnl3(in_ch, in_ch),
            ConvKnl3(in_ch, in_ch),
            ConvKnl3(in_ch, in_ch),
        )
    def forward(self, x):
        x = self.conv_stack(x)
        return x

class SepLargeConv(nn.Module):
    def __init__(self, in_ch):
        super(SepLargeConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(in_ch),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
            nn.BatchNorm3d(in_ch),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=(1, 1, 7), padding=(0, 0, 3)),
            nn.BatchNorm3d(in_ch),
            nn.ReLU()
        )
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        return out1, out2, out3

def pass_through(x):
    return x

class InputTransition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x16 = torch.cat((x,)*16, dim=1)
        out = F.relu(out + x16)
        return out

class DownTransition(nn.Module):
    """Block with mixed convolutions"""
    def __init__(self, in_ch, dropout=False):
        super(DownTransition, self).__init__()
        out_ch = 2 * in_ch
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

        self.drop = pass_through
        if dropout:
            self.drop = nn.Dropout3d()

        self.conv_stack = ConvStack(out_ch)
        self.sep_large_conv = SepLargeConv(out_ch)

    def __init__(self, x):
        down = self.down_conv(x)
        out = self.drop(down)

        out_conv_stack = self.conv_stack(out)
        out_lrg1, out_lrg2, out_lrg3 = self.sep_large_conv(out)

        out = out_conv_stack + out + out_lrg1 + out_lrg2 + out_lrg3
        out = F.relu(out)
        return out

class UpTransition(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch // 2),
            nn.ReLU()
        )

        self.do1 = pass_through
        if dropout:
            self.do1 = nn.Dropout3d()
        self.do2 = nn.Dropout3d()

        self.conv_stack = ConvStack(out_ch)

    def forward(self, x, skip):
        out = self.do1(x)
        skip = self.do2(skip)  # drop out
        out = self.up_conv(out)

        out_cat = torch.cat((out, skip), dim=1)  # e.g. 128 // 2 + 64 = 128
        out = self.conv_stack(out_cat)
        out = F.relu(out + out_cat)
        return out

class OutputTransition(nn.Module):
    def __init__(self, in_ch, out_ch=9):
        super(OutputTransition, self).__init__()
        self.conv1 = ConvKnl3(in_ch, out_ch)
        self.final_conv = nn.Conv3d(out_ch, out_ch, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.final_conv(out)
        return out

class FirstStageVNet(nn.Module):
    def __init__(self):
        super(FirstStageVNet, self).__init__()
        self.in_tr = InputTransition(1, 16)
        self.down_tr32 = DownTransition(16)
        self.down_tr64 = DownTransition(32)
        self.down_tr128 = DownTransition(64, dropout=True)
        self.down_tr256 = DownTransition(128, dropout=True)
        self.up_tr256 = UpTransition(256, 256, dropout=True)
        self.up_tr128 = UpTransition(256, 128, dropout=True)
        self.up_tr64 = UpTransition(128, 64)
        self.up_tr32 = UpTransition(64, 32)
        self.out_tr = OutputTransition(32)

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
        return out

class DownTransition(nn.Module):
    """Block with mixed convolutions"""
    def __init__(self, in_ch, dropout=False):
        super().__init__()
        out_ch = 2 * in_ch
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

        self.drop = pass_through
        if dropout:
            self.drop = nn.Dropout3d()

        self.conv_stack = ConvStack(out_ch)
        self.sep_large_conv = SepLargeConv(out_ch)

    def forward(self, x):
        down = self.down_conv(x)
        out = self.drop(down)

        out_conv_stack = self.conv_stack(out)
        out_lrg1, out_lrg2, out_lrg3 = self.sep_large_conv(out)

        out = out_conv_stack + out + out_lrg1 + out_lrg2 + out_lrg3
        out = F.relu(out)
        return out

if __name__ == '__main__':
    import os
    from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device('cuda')
    model = FirstStageVNet()
    model = model.to(dev)
    summary(model, (1, 160, 160, 64))
    # model = ConvStack(32)
    # model = model.to(dev)
    # summary(model, (1*32, 160//4, 160//4, 64//4))
