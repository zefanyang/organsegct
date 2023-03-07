#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/5/2021 5:44 PM
# @Author: yzf
from models.unet import *
from models.unet_edge_skip_con import EGModule, edge_fusion
from models.unet_rfp_uags import RFP_UAGs

class UNetFullScheme(nn.Module):
    def __init__(self, in_ch, out_ch, interpolate=True, conv_layer_order='cbr', init_ch=16):
        super(UNetFullScheme, self).__init__()

        self.no_class = out_ch

        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(2 * init_ch, 4 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(4 * init_ch, 8 * init_ch, conv_layer_order=conv_layer_order),
        ])

        self.decoders = nn.ModuleList([
            Decoder(4*init_ch+8*init_ch+32, 4*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(2*init_ch+4*init_ch+32, 2*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(1*init_ch+2*init_ch+32, init_ch, interpolate, conv_layer_order=conv_layer_order)
        ])

        self.rfp = RFP_UAGs(in_ch=8 * init_ch)

        self.edge_module = EGModule(init_ch)

        # self.final_conv3 = nn.Conv3d(4*init_ch, self.no_class, 1)
        # self.final_conv2 = nn.Conv3d(2*init_ch, self.no_class, 1)
        # self.final_conv1 = nn.Conv3d(init_ch, self.no_class, 1)

        self.final_conv = nn.Sequential(
            nn.Dropout3d(0.1),
            nn.Conv3d(init_ch, self.no_class, kernel_size=1))

    def forward(self, x):
        # encoder part
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        mid = self.encoders[3](enc3)
        encoders_features = [enc3, enc2, enc1]

        edge_feat, edge_score = self.edge_module(enc2, mid)

        ehn_mid = self.rfp(mid)

        skip3 = edge_fusion(enc3, edge_feat)
        skip2 = edge_fusion(enc2, edge_feat)
        skip1 = edge_fusion(enc1, edge_feat)

        dec3 = self.decoders[0](skip3, mid)
        dec2 = self.decoders[1](skip2, dec3)
        dec1 = self.decoders[2](skip1, dec2)

        # out3 = F.interpolate(self.final_conv3(dec3), scale_factor=4, mode='trilinear')
        # out2 = F.interpolate(self.final_conv2(dec2), scale_factor=2, mode='trilinear')
        # out1 = self.final_conv1(dec1)
        # score = out3 + out2 + out1

        score = self.final_conv(dec1)

        return score

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = UNetFullScheme(in_ch=1, out_ch=9).cuda()
    summary(model, (1, 160, 160, 64))