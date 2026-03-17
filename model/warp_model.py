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


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class WarpNet(nn.Module):
    """
    Predicts a 2-channel flow field to warp flat cloth onto the person.

    Input (25ch): agnostic(3) + pose(18) + cloth(3) + cloth_mask(1)
    Output:       flow (2, H/2, W/2) — upsampled in warp_cloth()
    """

    def __init__(self, in_channels=25):
        super().__init__()

        # Encoder  (H → H/2 → H/4 → H/8 → H/16)
        self.e1 = ConvBlock(in_channels, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)

        # Decoder with residual skip connections
        self.d1 = DeconvBlock(512, 256)    # + e3 → H/8
        self.d2 = DeconvBlock(256, 128)    # + e2 → H/4
        self.d3 = DeconvBlock(128, 64)     # + e1 → H/2

        # Flow head
        self.flow = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        d1 = self.d1(e4) + e3
        d2 = self.d2(d1) + e2
        d3 = self.d3(d2) + e1

        return self.flow(d3)
