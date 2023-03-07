#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/1/2021 12:24 PM
# @Author: yzf
"""Cascaded networks with mixed kernels described in TMI 2020 """
from models.cascaded_vnet.blsc_one_stage import *

class CombinationTransition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CombinationTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)

    def forward(self, x, seg_score):
        xcat = torch.cat((x, seg_score), dim=1)
        out = self.bn1(self.conv1(xcat))
        x16 = torch.cat((x,) * 16, dim=1)
        out = F.relu(out + x16)
        return out

class SecondStageVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_tr = CombinationTransition(1 + 9, 16)  # concatenate input and seg_score
        self.down_tr32 = DownTransition(16)
        self.down_tr64 = DownTransition(32)
        self.down_tr128 = DownTransition(64, dropout=True)
        self.down_tr256 = DownTransition(128, dropout=True)
        self.up_tr256 = UpTransition(256, 256, dropout=True)
        self.up_tr128 = UpTransition(256, 128, dropout=True)
        self.up_tr64 = UpTransition(128, 64)
        self.up_tr32 = UpTransition(64, 32)
        self.out_tr = OutputTransition(32)

    def forward(self, x, seg_score):
        out16 = self.in_tr(x, seg_score)
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

class CascadedNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = FirstStageVNet()
        self.net2 = SecondStageVNet()

    def forward(self, x):
        score1 = self.net1(x)
        score2 = self.net2(x, score1)
        return score1, score2

if __name__ == '__main__':
    import os
    from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device('cuda')
    model = CascadedNetworks()
    model = model.to(dev)
    summary(model, (1, 160, 160, 64))