"""
CP-VTON Style CNN U-Net — Direct Try-On Generation
====================================================
Architecture (CNN-Based, no explicit warping):

    Input Image (person + cloth)
         ↓
    [Encoder] DoubleConv + MaxPool2d
         ↓
    [Bottleneck] deepest features
         ↓
    [Decoder] Bilinear upsample + skip connections + DoubleConv
         ↓
    Output Image (try-on result)

Key operations:
    Conv2D → BatchNorm → ReLU  (double convolution per level)
    MaxPool2d(2)               to downsample
    Bilinear upsample          to reconstruct
    L1 loss                    to train

Input  (6ch):  person(3) + cloth(3) — concatenated directly, no warping
Output (3ch):  RGB try-on image in [-1, 1]

Reference: Han et al., "VITON" (CVPR 2018); Wang et al., "CP-VTON" (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv2D → BatchNorm → ReLU × 2 (standard CNN-Based building block)."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TryOnNet(nn.Module):
    """
    CP-VTON style CNN U-Net for direct virtual try-on.

    No explicit warping stage — the network learns to implicitly deform the
    cloth through its encoder-decoder features.

    Architecture: 4-level U-Net
        Encoder   : 4 × (DoubleConv → MaxPool2d)
        Bottleneck: DoubleConv at H/16
        Decoder   : 4 × (Bilinear×2 + cat skip + DoubleConv)
        Output    : Conv2d(ngf, 3) + tanh

    Input  (6ch) : person(3) + cloth(3)
    Output (3ch) : RGB try-on result in [-1, 1]

    Train with L1 loss.  VGG perceptual loss optional for texture quality.
    """

    def __init__(self, in_channels: int = 6, ngf: int = 64):
        super().__init__()
        nf = ngf

        # ── Encoder: Conv+BN+ReLU blocks + MaxPool ─────────────────────────
        self.enc1 = DoubleConv(in_channels, nf)       # H   × W   → nf
        self.enc2 = DoubleConv(nf,     nf * 2)        # H/2 × W/2 → nf*2
        self.enc3 = DoubleConv(nf * 2, nf * 4)        # H/4 × W/4 → nf*4
        self.enc4 = DoubleConv(nf * 4, nf * 8)        # H/8 × W/8 → nf*8
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ──────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(nf * 8, nf * 16)  # H/16 → nf*16

        # ── Decoder: bilinear upsample + concat skip + DoubleConv ──────────
        self.up4  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(nf * 16 + nf * 8, nf * 8)   # H/8

        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(nf * 8  + nf * 4, nf * 4)   # H/4

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(nf * 4  + nf * 2, nf * 2)   # H/2

        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(nf * 2  + nf,     nf)        # H

        # ── Output head ─────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(nf, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, H, W)  — torch.cat([person, cloth], dim=1)
        returns: (B, 3, H, W) in [-1, 1]
        """
        # Encoder — downsample with MaxPool
        e1 = self.enc1(x)                    # H,   nf
        e2 = self.enc2(self.pool(e1))        # H/2, nf*2
        e3 = self.enc3(self.pool(e2))        # H/4, nf*4
        e4 = self.enc4(self.pool(e3))        # H/8, nf*8

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))  # H/16, nf*16

        # Decoder — bilinear upsample + skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))  # H/8
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # H/4
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # H/2
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # H

        return torch.tanh(self.out_conv(d1))
