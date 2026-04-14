"""
TryOnNet V2 — Composition-based Try-On Generator
==================================================
Instead of directly generating the entire output image, predicts:
  1. A rendered person region (face, arms, legs, background)
  2. A composition mask alpha ∈ [0, 1]
  3. Final = alpha * warped_cloth + (1 - alpha) * rendered

This forces the network to preserve the warped cloth texture exactly
in the clothing region, instead of hallucinating/blurring it.

Architecture: 5-block U-Net encoder-decoder with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with InstanceNorm — preserves spatial detail."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + concat skip + double conv."""

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TryOnNetV2(nn.Module):
    """
    Composition-based TryOnNet.

    Input  (25ch): agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18)
    Output: final try-on image (3ch, [-1, 1])

    Internally predicts:
      - rendered: person region (non-cloth areas)
      - alpha: composition mask (where to use warped cloth)
      - output = alpha * warped_cloth + (1 - alpha) * rendered
    """

    def __init__(self, in_channels=25, ngf=64):
        super().__init__()

        # Encoder (5 levels)
        self.e1 = DownBlock(in_channels, ngf)         # H/2
        self.e2 = DownBlock(ngf, ngf * 2)             # H/4
        self.e3 = DownBlock(ngf * 2, ngf * 4)         # H/8
        self.e4 = DownBlock(ngf * 4, ngf * 8)         # H/16
        self.e5 = DownBlock(ngf * 8, ngf * 8)         # H/32

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(ngf * 8),
            ResBlock(ngf * 8),
        )

        # Decoder with skip connections
        self.d5 = UpBlock(ngf * 8, ngf * 8, ngf * 8)   # + e4
        self.d4 = UpBlock(ngf * 8, ngf * 4, ngf * 4)   # + e3
        self.d3 = UpBlock(ngf * 4, ngf * 2, ngf * 2)   # + e2
        self.d2 = UpBlock(ngf * 2, ngf, ngf)            # + e1

        # Final upsample to full resolution
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # Two output heads
        self.render_head = nn.Conv2d(ngf, 3, 3, 1, 1)   # rendered person
        self.alpha_head = nn.Conv2d(ngf, 1, 3, 1, 1)     # composition mask

    def forward(self, x, warped_cloth=None):
        """
        Args:
            x: (B, 25, H, W) concatenated input
            warped_cloth: (B, 3, H, W) warped cloth for composition.
                          If None, returns rendered output directly.
        """
        # Encoder
        e1 = self.e1(x)    # H/2
        e2 = self.e2(e1)   # H/4
        e3 = self.e3(e2)   # H/8
        e4 = self.e4(e3)   # H/16
        e5 = self.e5(e4)   # H/32

        # Bottleneck
        b = self.bottleneck(e5)

        # Decoder
        d5 = self.d5(b, e4)    # H/16
        d4 = self.d4(d5, e3)   # H/8
        d3 = self.d3(d4, e2)   # H/4
        d2 = self.d2(d3, e1)   # H/2

        # Full resolution
        feat = self.up_final(d2)  # H

        # Output heads
        rendered = torch.tanh(self.render_head(feat))     # [-1, 1]
        alpha = torch.sigmoid(self.alpha_head(feat))       # [0, 1]

        if warped_cloth is not None:
            output = alpha * warped_cloth + (1 - alpha) * rendered
        else:
            output = rendered

        return output, rendered, alpha
