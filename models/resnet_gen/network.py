"""
ResNet-style generator (pix2pix ResNet9) for virtual try-on.

No skip connections — uses residual blocks in the bottleneck to learn
global cloth-to-person mapping via style transfer.

Input:  25ch  (agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18))
Output: 3ch RGB in [-1, 1]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from model.warp_model import WarpNet  # noqa: F401  (re-exported for convenience)


class ResBlock(nn.Module):
    """
    Residual block with reflection padding.
    ReflectionPad → Conv3 → IN → ReLU → ReflectionPad → Conv3 → IN  + residual.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    pix2pix ResNet9 generator adapted for 25-channel try-on input.

    Architecture:
      ReflectionPad(3) → Conv7 → IN → ReLU
      → 2× (Conv3 stride-2 → IN → ReLU)          [downsampling]
      → 9× ResBlock                                [transformation]
      → 2× (ConvTranspose3 stride-2 → IN → ReLU)  [upsampling]
      → ReflectionPad(3) → Conv7 → Tanh

    in_channels : 25
    ngf         : 64
    n_blocks    : 9
    """

    def __init__(self, in_channels: int = 25, ngf: int = 64, n_blocks: int = 9):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling × 2
        for i in range(2):
            mult = 2 ** i
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # Residual blocks
        mult = 2 ** 2  # = 4  (channels = ngf * 4)
        for _ in range(n_blocks):
            layers.append(ResBlock(ngf * mult))

        # Upsampling × 2
        for i in range(2):
            mult = 2 ** (2 - i)
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
                ),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]

        # Output head
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
