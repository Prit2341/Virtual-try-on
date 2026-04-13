"""
CP-VITON — Characteristic-Preserving Virtual Try-On Network
=============================================================
Wang et al., "Toward Characteristic-Preserving Image-based Virtual
Try-On Network" (ECCV 2018) + PatchGAN adversarial training.

Pipeline
--------
  Stage 1 — GMM (Geometric Matching Module)
    TPS warp: cloth + cloth_mask + agnostic + pose → warped_cloth, warped_mask
    (reuses GMMNet from model/gmm_model.py)

  Stage 2 — TOM (Try-On Module) + PatchGAN
    U-Net: agnostic + warped_cloth + warped_mask + pose → rendered + alpha
    final = alpha * warped_cloth + (1 - alpha) * rendered
    PatchGAN discriminator: enforces photo-realistic outputs

Key improvements over v2
------------------------
  * PatchGAN discriminator → sharper, more realistic textures
  * Feature matching loss → stable GAN training
  * Stronger cloth-mask consistency loss
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the proven GMM from model/
from model.gmm_model import GMMNet  # noqa: F401  (re-exported for convenience)


# ---------------------------------------------------------------------------
# TOM — Try-On Module
# ---------------------------------------------------------------------------

class _DownBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, dropout: float = 0.0):
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
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return self.drop(self.conv(torch.cat([x, skip], dim=1)))


class TryOnModule(nn.Module):
    """
    CP-VITON Try-On Module (TOM).

    A 5-level U-Net that takes the agnostic person representation and
    warped cloth, then produces:
      - rendered  (3ch, tanh)  : person body without any clothing
      - alpha     (1ch, sigmoid): blending mask (1=warped cloth, 0=rendered)
      - output    (3ch)        : alpha * warped_cloth + (1-alpha) * rendered

    Input  : agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18) = 25ch
    """

    def __init__(self, in_channels: int = 25, ngf: int = 64):
        super().__init__()

        # Encoder
        self.e1 = _DownBlock(in_channels, ngf)         # H/2
        self.e2 = _DownBlock(ngf,         ngf * 2)     # H/4
        self.e3 = _DownBlock(ngf * 2,     ngf * 4)     # H/8
        self.e4 = _DownBlock(ngf * 4,     ngf * 8)     # H/16
        self.e5 = _DownBlock(ngf * 8,     ngf * 8)     # H/32

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.d5 = _UpBlock(ngf * 8, ngf * 8, ngf * 8, dropout=0.3)
        self.d4 = _UpBlock(ngf * 8, ngf * 4, ngf * 4)
        self.d3 = _UpBlock(ngf * 4, ngf * 2, ngf * 2)
        self.d2 = _UpBlock(ngf * 2, ngf,     ngf)

        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        self.render_head = nn.Conv2d(ngf, 3, 3, 1, 1)
        self.alpha_head  = nn.Conv2d(ngf, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor, warped_cloth: torch.Tensor):
        """
        Args:
            x            : (B, 25, H, W) concatenated inputs
            warped_cloth : (B,  3, H, W) TPS-warped cloth for composition

        Returns:
            output   (B, 3, H, W)  final try-on image
            rendered (B, 3, H, W)  rendered person (no cloth)
            alpha    (B, 1, H, W)  composition mask
        """
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        b  = self.bottleneck(e5)

        d5 = self.d5(b,  e4)
        d4 = self.d4(d5, e3)
        d3 = self.d3(d4, e2)
        d2 = self.d2(d3, e1)

        feat     = self.up_final(d2)
        rendered = torch.tanh(self.render_head(feat))
        alpha    = torch.sigmoid(self.alpha_head(feat))
        output   = alpha * warped_cloth + (1.0 - alpha) * rendered

        return output, rendered, alpha


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchGAN(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Classifies overlapping 70×70 image patches as real or fake, giving
    dense supervision rather than a single scalar. Each output unit
    corresponds to a receptive field of ~70×70 pixels.

    Input: image (3ch) concatenated with condition (agnostic+warped_cloth = 6ch)
           total = 9ch
    Output: (B, 1, H/16, W/16) — patch real/fake predictions
    """

    def __init__(self, in_channels: int = 9, ndf: int = 64, n_layers: int = 3):
        super().__init__()

        # First layer: no norm
        layers = [
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final layers: no downsampling
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, 1, 1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Feature Matching Loss helper
# ---------------------------------------------------------------------------

class FeatureMatchingLoss(nn.Module):
    """
    Extracts intermediate discriminator features for both real and fake,
    and penalises their L1 distance. Stabilises GAN training significantly.
    """

    def __init__(self, discriminator: PatchGAN, n_layers: int = 3):
        super().__init__()
        # Split discriminator into per-layer feature extractors
        self.layers = nn.ModuleList()
        seq = list(discriminator.model.children())
        # First block (2 ops: conv + lrelu)
        self.layers.append(nn.Sequential(*seq[:2]))
        idx = 2
        for _ in range(n_layers - 1):
            self.layers.append(nn.Sequential(*seq[idx:idx + 3]))
            idx += 3
        # Remaining
        self.layers.append(nn.Sequential(*seq[idx:]))

    def forward(self, fake_cond: torch.Tensor,
                real_cond: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=fake_cond.device)
        f, r = fake_cond, real_cond
        for layer in self.layers:
            f = layer(f)
            r = layer(r)
            loss = loss + F.l1_loss(f, r.detach())
        return loss / len(self.layers)
