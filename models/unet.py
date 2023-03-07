import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate=True, conv_layer_order='cbr', init_ch=16):
        super(UNet3D, self).__init__()

        self.no_class = out_channels
        # number of groups for the GroupNorm
        # num_groups = min(init_ch // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(2 * init_ch, 4 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(4 * init_ch, 8 * init_ch, conv_layer_order=conv_layer_order),
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_ch + 8 * init_ch, 4 * init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(2 * init_ch + 4 * init_ch, 2 * init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(init_ch + 2 * init_ch, init_ch, interpolate, conv_layer_order=conv_layer_order)
        ])

        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(init_ch, self.no_class, 1))

    def forward(self, x):
        # encoder part
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        mid = self.encoders[3](enc3)
        encoders_features = [enc3, enc2, enc1]

        dec3 = self.decoders[0](enc3, mid)
        dec2 = self.decoders[1](enc2, dec3)
        dec1 = self.decoders[2](enc1, dec2)

        final = self.final_conv(dec1)
        return final

# Some correctly implemented utilities from a github code repository,
# but I don't like them.
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, is_max_pool=True,
                 max_pool_kernel_size=(2, 2, 2), conv_layer_order='cbr', num_groups=32):
        super(Encoder, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, padding=0) if is_max_pool else None
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=conv_kernel_size,
                                      order=conv_layer_order,
                                      num_groups=num_groups)

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, interpolate, kernel_size=3,
                 scale_factor=(2, 2, 2), conv_layer_order='cbr', num_groups=32):
        super(Decoder, self).__init__()
        if interpolate:
            self.upsample = None
        else:
            self.upsample = nn.ConvTranspose3d(2 * out_channels,
                                               2 * out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=0)
        self.double_conv = DoubleConv(in_channels, out_channels,
                                      kernel_size=kernel_size,
                                      order=conv_layer_order,
                                      num_groups=num_groups)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='trilinear')
        else:
            x = self.upsample(x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cbr', num_groups=32):
        super(DoubleConv, self).__init__()
        if in_channels < out_channels:
            # if in_channels < out_channels we're in the encoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # otherwise we're in the decoder path
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self._add_conv(1, conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups)
        # conv2
        self._add_conv(2, conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups)

    def _add_conv(self, pos, in_channels, out_channels, kernel_size, order, num_groups):
        assert pos in [1, 2], 'pos MUST be either 1 or 2'
        assert 'c' in order, "'c' (conv layer) MUST be present"
        assert 'r' in order, "'r' (ReLU layer) MUST be present"

        for i, char in enumerate(order):
            if char == 'r':
                self.add_module(f'relu{pos}', nn.ReLU(inplace=True))
            elif char == 'c':
                self.add_module(f'conv{pos}', nn.Conv3d(in_channels,
                                                        out_channels,
                                                        kernel_size,
                                                        padding=1))
            elif char == 'g':
                is_before_conv = i < order.index('c')
                assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
                self.add_module(f'norm{pos}', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
            elif char == 'b':
                is_before_conv = i < order.index('c')
                if is_before_conv:
                    self.add_module(f'norm{pos}', nn.BatchNorm3d(in_channels))
                else:
                    self.add_module(f'norm{pos}', nn.BatchNorm3d(out_channels))
            else:
                raise ValueError(
                    f"Unsupported layer type '{char}'. MUST be one of 'b', 'r', 'c'")


if __name__ == '__main__':
    import time
    model = UNet3D(1, 9, init_ch=16, conv_layer_order='cbr', interpolate=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    start = time.time()
    summary(model, (1, 160, 160, 64))
    print("take {:f} s".format(time.time() - start))