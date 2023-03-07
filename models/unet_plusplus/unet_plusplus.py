#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/5/2021 2:47 PM
# @Author: yzf
from models.unet import *

def cat(*x):
    return torch.cat(x, dim=1)

class UNetPlusPlus(nn.Module):
    """UNet++ with depth of 4"""
    def __init__(self, no_class=9):
        super().__init__()
        filter_ch = (16, 32, 64, 128, 256)

        # enc_ij, where i indexes down-sampling layer
        # and j indexes skip connection layer.
        self.enc_00 = Encoder(1, filter_ch[0], is_max_pool=False)  # 1 -> 16
        self.enc_10 = Encoder(filter_ch[0], filter_ch[1])  # 16 -> 32
        self.enc_20 = Encoder(filter_ch[1], filter_ch[2])  # 32 -> 64
        self.enc_30 = Encoder(filter_ch[2], filter_ch[3])  # 64 -> 128
        self.enc_40 = Encoder(filter_ch[3], filter_ch[4])  # 128 -> 256

        self.dec_01 = Decoder(filter_ch[0]+filter_ch[1], filter_ch[0], interpolate=True)  # (16+32) -> 16

        self.dec_11 = Decoder(filter_ch[1]+filter_ch[2], filter_ch[1], interpolate=True)  # (32+64) -> 32
        self.dec_02 = Decoder(2*filter_ch[0]+filter_ch[1], filter_ch[0], interpolate=True)  # (2*16+32) -> 16

        self.dec_21 = Decoder(filter_ch[2]+filter_ch[3], filter_ch[2], interpolate=True)
        self.dec_12 = Decoder(2*filter_ch[1]+filter_ch[2], filter_ch[1], interpolate=True)
        self.dec_03 = Decoder(3*filter_ch[0]+filter_ch[1], filter_ch[0], interpolate=True)

        self.dec_31 = Decoder(filter_ch[3]+filter_ch[4], filter_ch[3], interpolate=True)
        self.dec_22 = Decoder(2*filter_ch[2]+filter_ch[3], filter_ch[2], interpolate=True)
        self.dec_13 = Decoder(3*filter_ch[1]+filter_ch[2], filter_ch[1], interpolate=True)
        self.dec_04 = Decoder(4*filter_ch[0]+filter_ch[1], filter_ch[0], interpolate=True)

        # self.fnl1 = nn.Conv3d(filter_ch[0], no_class, kernel_size=1)
        # self.fnl2 = nn.Conv3d(filter_ch[0], no_class, kernel_size=1)
        # self.fnl3 = nn.Conv3d(filter_ch[0], no_class, kernel_size=1)
        self.fnl4 = nn.Conv3d(filter_ch[0], no_class, kernel_size=1)


    def forward(self, x):
        x00 = self.enc_00(x)
        x10 = self.enc_10(x00)
        x20 = self.enc_20(x10)
        x30 = self.enc_30(x20)
        x40 = self.enc_40(x30)

        x01 = self.dec_01(x00, x10)  # enc_feat, x

        x11 = self.dec_11(x10, x20)
        x02 = self.dec_02(cat(x00, x01), x11)

        x21 = self.dec_21(x20, x30)
        x12 = self.dec_12(cat(x10, x11), x21)
        x03 = self.dec_03(cat(x00, x01, x02), x12)

        x31 = self.dec_31(x30, x40)
        x22 = self.dec_22(cat(x20, x21), x31)
        x13 = self.dec_13(cat(x10, x11, x12), x22)
        x04 = self.dec_04(cat(x00, x01, x02, x03), x13)

        # out1 = self.fnl1(x01)
        # out2 = self.fnl2(x02)
        # out3 = self.fnl3(x03)
        out4 = self.fnl4(x04)

        # Adding the outputs could hinder the learning process.
        # Simply supervise the final output may be better.
        # score = out1 + out2 + out3 + out4

        return out4

class UNetPlusPlusL3(nn.Module):
    """UNet++ with depth of 3"""
    def __init__(self, no_class=9):
        super().__init__()
        filter_ch = (16, 32, 64, 128)

        # enc_ij, where i indexes down-sampling layer
        # and j indexes skip connection layer.
        self.enc_00 = Encoder(1, filter_ch[0], is_max_pool=False)  # 1 -> 16
        self.enc_10 = Encoder(filter_ch[0], filter_ch[1])  # 16 -> 32
        self.enc_20 = Encoder(filter_ch[1], filter_ch[2])  # 32 -> 64
        self.enc_30 = Encoder(filter_ch[2], filter_ch[3])  # 64 -> 128

        self.dec_01 = Decoder(filter_ch[0]+filter_ch[1], filter_ch[0], interpolate=True)  # (16+32) -> 16

        self.dec_11 = Decoder(filter_ch[1]+filter_ch[2], filter_ch[1], interpolate=True)  # (32+64) -> 32
        self.dec_02 = Decoder(2*filter_ch[0]+filter_ch[1], filter_ch[0], interpolate=True)  # (2*16+32) -> 16

        self.dec_21 = Decoder(filter_ch[2]+filter_ch[3], filter_ch[2], interpolate=True)
        self.dec_12 = Decoder(2*filter_ch[1]+filter_ch[2], filter_ch[1], interpolate=True)
        self.dec_03 = Decoder(3*filter_ch[0]+filter_ch[1], filter_ch[0], interpolate=True)

        # self.fnl1 = nn.Conv3d(filter_ch[0], no_class, kernel_size=1)
        # self.fnl2 = nn.Conv3d(filter_ch[0], no_class, kernel_size=1)
        self.fnl3 = nn.Conv3d(filter_ch[0], no_class, kernel_size=1)


    def forward(self, x):
        x00 = self.enc_00(x)
        x10 = self.enc_10(x00)
        x20 = self.enc_20(x10)
        x30 = self.enc_30(x20)

        x01 = self.dec_01(x00, x10)  # enc_feat, x

        x11 = self.dec_11(x10, x20)
        x02 = self.dec_02(cat(x00, x01), x11)

        x21 = self.dec_21(x20, x30)
        x12 = self.dec_12(cat(x10, x11), x21)
        x03 = self.dec_03(cat(x00, x01, x02), x12)

        # out1 = self.fnl1(x01)
        # out2 = self.fnl2(x02)
        out3 = self.fnl3(x03)

        # score = out1 + out2 + out3

        return out3

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = UNetPlusPlus().cuda()

    from models.unet_nine_layers.unet_l9 import count_parameters
    print('Total number of trainable parameters: {:.2f} M'.format(count_parameters(model) / 1e6))