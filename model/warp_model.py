import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample → concat skip → conv."""

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c, 4, 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], 1)
        return self.conv(x)


class WarpNet(nn.Module):
    """
    Lightweight WarpNet — predicts 2ch flow field for cloth warping.

    Input  (25ch): agnostic(3) + pose(18) + cloth(3) + cloth_mask(1)
    Output (2ch):  flow field at H/2 resolution (upsampled in warp_cloth)

    4-block U-Net with concat skips.
    flow_scale controls max displacement as fraction of image size.
    """

    def __init__(self, in_channels=25, ngf=64, flow_scale=0.5):
        super().__init__()
        self.flow_scale = flow_scale

        # Encoder  (H → H/2 → H/4 → H/8 → H/16)
        self.e1 = ConvBlock(in_channels, ngf)
        self.e2 = ConvBlock(ngf,     ngf * 2)
        self.e3 = ConvBlock(ngf * 2, ngf * 4)
        self.e4 = ConvBlock(ngf * 4, ngf * 8)

        # Decoder with concat skip connections
        self.d1 = UpBlock(ngf * 8, ngf * 4, ngf * 4)
        self.d2 = UpBlock(ngf * 4, ngf * 2, ngf * 2)
        self.d3 = UpBlock(ngf * 2, ngf,     ngf)

        # Flow head — output at H/2
        self.flow = nn.Conv2d(ngf, 2, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)   # H/2
        e2 = self.e2(e1)  # H/4
        e3 = self.e3(e2)  # H/8
        e4 = self.e4(e3)  # H/16

        d1 = self.d1(e4, e3)  # H/8
        d2 = self.d2(d1, e2)  # H/4
        d3 = self.d3(d2, e1)  # H/2

        return torch.tanh(self.flow(d3)) * self.flow_scale
