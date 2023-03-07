#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/29/2021 3:07 PM
# @Author: yzf
"""Paper version V-Net architecture with 3*3*3 convolution kernels"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        # we change the kernel size to 3
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        # self.bn1 = ContBatchNorm3d(16)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        # self.bn1 = ContBatchNorm3d(outChans)
        self.bn1 = nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        # down conv
        down = self.relu1(self.bn1(self.down_conv(x)))
        # drop out or pass through
        out = self.do1(down)
        # n convolutional units
        out = self.ops(out)
        # add, and then ReLU to ensure non-linearity
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        # self.bn1 = ContBatchNorm3d(outChans // 2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)

        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        # drop out or pass through
        out = self.do1(x)
        # skip connection with drop out
        skipxdo = self.do2(skipx)
        # up conv and half the channels
        out = self.relu1(self.bn1(self.up_conv(out)))
        # concatenation
        xcat = torch.cat((out, skipxdo), 1)
        # n convolutional units
        out = self.ops(xcat)
        # add, and then ReLU to ensure non-linearity
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 9, kernel_size=3, padding=1)
        # self.bn1 = ContBatchNorm3d(2)
        self.bn1 = nn.BatchNorm3d(9)

        self.conv2 = nn.Conv3d(9, 9, kernel_size=1)
        self.relu1 = ELUCons(elu, 9)
        # if nll:
        #     self.softmax = F.log_softmax
        # else:
        #     self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 9 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out

class VNetKnl3PaperVer(nn.Module):
    # The network topology as described in the diagram
    # in the VNet paper
    def __init__(self, elu=True, nll=False):
        super(VNetKnl3PaperVer, self).__init__()
        self.in_tr =  InputTransition(16, elu)
        # the number of convolutions in each layer corresponds
        # to what is in the actual prototxt, not the intent
        self.down_tr32 = DownTransition(16, 2, elu)
        self.down_tr64 = DownTransition(32, 3, elu)
        self.down_tr128 = DownTransition(64, 3, elu)
        self.down_tr256 = DownTransition(128, 3, elu)
        self.up_tr256 = UpTransition(256, 256, 3, elu)
        self.up_tr128 = UpTransition(256, 128, 3, elu)
        self.up_tr64 = UpTransition(128, 64, 2, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

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

if __name__ == '__main__':
    import os
    import time
    from torchsummary import summary
    model = VNetKnl3PaperVer()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    start = time.time()
    summary(model, (1, 160, 160, 64))
    print("take {:f} s".format(time.time() - start))