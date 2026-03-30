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


class TryOnNet(nn.Module):
    """
    Lightweight TryOnNet — synthesizes final try-on RGB image.

    Input  (25ch): agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18)
    Output (3ch):  RGB image in [-1, 1]

    4-block U-Net with concat skips + final full-res upsample, ngf=32.
    Converges in 5-10 epochs on ~10k samples.
    """

    def __init__(self, in_channels=25, ngf=64):
        super().__init__()
        # Encoder  (H → H/2 → H/4 → H/8 → H/16)
        self.e1 = ConvBlock(in_channels, ngf)        # 32
        self.e2 = ConvBlock(ngf,     ngf * 2)        # 64
        self.e3 = ConvBlock(ngf * 2, ngf * 4)        # 128
        self.e4 = ConvBlock(ngf * 4, ngf * 8)        # 256  (bottleneck)

        # Decoder with concat skip connections
        self.d1 = UpBlock(ngf * 8, ngf * 4, ngf * 4)  # + e3 → 128
        self.d2 = UpBlock(ngf * 4, ngf * 2, ngf * 2)  # + e2 → 64
        self.d3 = UpBlock(ngf * 2, ngf,     ngf)       # + e1 → 32

        # Final upsample to full resolution
        self.up_final = nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1)  # → 16ch at H
        self.out = nn.Conv2d(ngf // 2, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)   # H/2
        e2 = self.e2(e1)  # H/4
        e3 = self.e3(e2)  # H/8
        e4 = self.e4(e3)  # H/16

        d1 = self.d1(e4, e3)  # H/8
        d2 = self.d2(d1, e2)  # H/4
        d3 = self.d3(d2, e1)  # H/2

        out = self.up_final(d3)  # H
        return torch.tanh(self.out(out))
