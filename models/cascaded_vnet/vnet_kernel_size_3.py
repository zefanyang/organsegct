#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/29/2021 10:31 AM
# @Author: yzf
"""Modified from https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

def non_linearity(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = non_linearity(elu, nchan)
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
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = non_linearity(elu, 16)

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
        self.bn1 = nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = non_linearity(elu, outChans)
        self.relu2 = non_linearity(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)

        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = non_linearity(elu, outChans // 2)
        self.relu2 = non_linearity(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))  # double resolution, half channels, e.g., 256 -> 128
        xcat = torch.cat((out, skipxdo), 1)  # 128 cat 128 -> 256
        out = self.ops(xcat)  # 256 -> 256
        out = self.relu2(torch.add(out, xcat))  # shortcut connection
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 9, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(9)

        self.conv2 = nn.Conv3d(9, 9, kernel_size=1)
        self.relu1 = non_linearity(elu, 9)
        # if nll:
        #     self.softmax = F.log_softmax
        # else:
        #     self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNetKnl3(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True):
        super(VNetKnl3, self).__init__()
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
        return out

if __name__ == '__main__':
    import os
    import time
    from torchsummary import summary
    model = VNetKnl3()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    from models.unet_nine_layers.unet_l9 import count_parameters, parameter_table
    print(parameter_table(model))
    print('Total number of trainable parameters: {:.2f} M'.format(count_parameters(model) / 1e6))

# import torch.nn as nn
#
# class VNet(nn.Module):
#     def __init__(self):
#         pass
# This is merely block with residual connection
# class res_block(nn.Module):
#     def __init__(self, in_ch, out_ch, is_pool=True):
#         self.db_conv = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, 1, 1),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(),
#             nn.Conv3d(out_ch, out_ch, 3, 1, 1),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(),
#         )
#         self.max_pool = nn.MaxPool3d(kernel_size=2, padding=0) if is_pool else None
#
#     def forward(self, x):
#         if self.max_pool is not None:
#             x = self.max_pool(x)
#         x = self.db_conv(x)


# normalization between sub-volumes is necessary
# for good performance
# class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
#     def _check_input_dim(self, input):
#         if input.dim() != 5:
#             raise ValueError('expected 5D input (got {}D input)'
#                              .format(input.dim()))
#         # super(ContBatchNorm3d, self)._check_input_dim(input)
#
#     def forward(self, input):
#         self._check_input_dim(input)
#         return F.batch_norm(
#             input, self.running_mean, self.running_var, self.weight, self.bias,
#             True, self.momentum, self.eps)