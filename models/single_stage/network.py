"""
Single-stage virtual try-on: deep 5-level U-Net with no warping stage.

The network directly maps (agnostic, cloth, cloth_mask, pose) → RGB person image,
relying on the U-Net's receptive field to learn implicit cloth alignment.

Input:  25ch  (agnostic(3) + cloth(3) + cloth_mask(1) + pose(18))
Output: 3ch RGB in [-1, 1]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample → concat skip → conv."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c, 4, 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SingleStageTryOn(nn.Module):
    """
    Deep 5-level U-Net for single-stage virtual try-on.

    Input  (25ch): agnostic(3) + cloth(3) + cloth_mask(1) + pose(18)
    Output (3ch):  RGB image in [-1, 1]

    Encoder:
      e1: H   → H/2   (25  → ngf)
      e2: H/2 → H/4   (ngf → ngf*2)
      e3: H/4 → H/8   (ngf*2 → ngf*4)
      e4: H/8 → H/16  (ngf*4 → ngf*8)
      e5: H/16→ H/32  (ngf*8 → ngf*8)

    Decoder (with skip connections):
      d1: H/32 → H/16  skip=e4
      d2: H/16 → H/8   skip=e3
      d3: H/8  → H/4   skip=e2
      d4: H/4  → H/2   skip=e1
      up_final: H/2 → H
      out: conv → Tanh
    """

    def __init__(self, in_channels: int = 25, ngf: int = 64):
        super().__init__()

        # Encoder
        self.e1 = ConvBlock(in_channels, ngf)        # H/2
        self.e2 = ConvBlock(ngf,         ngf * 2)    # H/4
        self.e3 = ConvBlock(ngf * 2,     ngf * 4)    # H/8
        self.e4 = ConvBlock(ngf * 4,     ngf * 8)    # H/16
        self.e5 = ConvBlock(ngf * 8,     ngf * 8)    # H/32

        # Decoder
        self.d1 = UpBlock(ngf * 8, ngf * 8, ngf * 8)  # skip=e4
        self.d2 = UpBlock(ngf * 8, ngf * 4, ngf * 4)  # skip=e3
        self.d3 = UpBlock(ngf * 4, ngf * 2, ngf * 2)  # skip=e2
        self.d4 = UpBlock(ngf * 2, ngf,     ngf)       # skip=e1

        # Final upsample H/2 → H
        self.up_final = nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1)
        self.out      = nn.Conv2d(ngf // 2, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)   # H/2
        e2 = self.e2(e1)  # H/4
        e3 = self.e3(e2)  # H/8
        e4 = self.e4(e3)  # H/16
        e5 = self.e5(e4)  # H/32

        d1 = self.d1(e5, e4)  # H/16
        d2 = self.d2(d1, e3)  # H/8
        d3 = self.d3(d2, e2)  # H/4
        d4 = self.d4(d3, e1)  # H/2

        out = self.up_final(d4)          # H
        return torch.tanh(self.out(out))
