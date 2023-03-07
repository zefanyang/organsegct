#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/1/2021 12:24 PM
# @Author: yzf
"""Cascaded V-Net: only concatenate final output"""
from models.cascaded_vnet.vnet_kernel_size_3 import *

class CombinationTransition(nn.Module):
    """Concatenate final output"""
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

class SecondStageVNet_FnlOutput(nn.Module):
    """Second stage network, which incorporates the decoder features from the fist stage network."""
    def __init__(self, elu=True):
        super(SecondStageVNet_FnlOutput, self).__init__()
        self.in_tr = CombinationTransition(1 + 9, 16)  # concatenate input and seg_score
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

class CascadedNetworksFnlOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = VNetKnl3()
        self.net2 = SecondStageVNet_FnlOutput()

    def forward(self, x):
        score1 = self.net1(x)
        score2 = self.net2(x, score1)
        return score1, score2

if __name__ == '__main__':
    import os
    from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device('cuda')
    model = CascadedNetworksFnlOutput()
    model = model.to(dev)

    from models.unet_nine_layers.unet_l9 import count_parameters
    print('Total number of trainable parameters: {:.2f} M'.format(count_parameters(model) / 1e6))