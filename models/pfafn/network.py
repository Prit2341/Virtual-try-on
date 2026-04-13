"""
PF-AFN — Parser-Free Appearance Flow Network
=============================================
Inspired by: "Parser-Free Virtual Try-on via Distilling Appearance Flows"
             (CVPR 2021, Ge et al.)

Key idea
--------
Instead of TPS (Thin Plate Spline) with sparse control points, PF-AFN
estimates a DENSE appearance flow field directly from cloth and agnostic
person features. This gives:
  * More flexible local deformations than TPS
  * No human parsing (segmentation) required at inference time
  * Better handling of complex folds and textures

Architecture
------------
  AppearanceFlowNet (AFN)
    Encoder: VGG-style feature pyramid for both cloth and agnostic
    Correlation: multi-scale cross-attention between cloth and body features
    Decoder: upsamples correlation to dense flow field (2ch per level)
    Final: grid_sample(cloth, flow) → warped cloth

  ContentFusionNet (CFN)
    U-Net synthesis: agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18)
                     → person output (3ch)

Inputs (parser-free at inference)
----------------------------------
  cloth       (B, 3, H, W)   product clothing image
  cloth_mask  (B, 1, H, W)   binary cloth mask
  agnostic    (B, 3, H, W)   person with clothing region erased
  pose        (B, 18, H, W)  OpenPose/MediaPipe heatmaps
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ConvNormRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1,
                 norm: str = "instance", activation: str = "relu"):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)]
        if norm == "instance":
            layers.append(nn.InstanceNorm2d(out_c))
        elif norm == "batch":
            layers.append(nn.BatchNorm2d(out_c))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Appearance Flow Network (AFN)
# ---------------------------------------------------------------------------

class _FeaturePyramid(nn.Module):
    """
    4-level feature pyramid encoder.
    Returns features at H/2, H/4, H/8, H/16.
    """

    def __init__(self, in_c: int, ngf: int = 64):
        super().__init__()
        self.l1 = DownBlock(in_c,      ngf)       # H/2,  ngf
        self.l2 = DownBlock(ngf,       ngf * 2)   # H/4,  ngf*2
        self.l3 = DownBlock(ngf * 2,   ngf * 4)   # H/8,  ngf*4
        self.l4 = DownBlock(ngf * 4,   ngf * 8)   # H/16, ngf*8

    def forward(self, x):
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return f1, f2, f3, f4


class _CrossCorrelation(nn.Module):
    """
    Lightweight cross-correlation between cloth and body feature maps.
    Computes channel-wise dot product and returns correlation features.
    """

    def __init__(self, feat_c: int, out_c: int):
        super().__init__()
        self.proj_cloth  = nn.Conv2d(feat_c, out_c, 1, bias=False)
        self.proj_body   = nn.Conv2d(feat_c, out_c, 1, bias=False)
        self.fuse        = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, cloth_feat: torch.Tensor,
                body_feat: torch.Tensor) -> torch.Tensor:
        c = F.normalize(self.proj_cloth(cloth_feat), dim=1)
        b = F.normalize(self.proj_body(body_feat),   dim=1)
        corr = c * b   # element-wise product after normalization
        return self.fuse(corr)


class AppearanceFlowNet(nn.Module):
    """
    Dense appearance flow estimator.

    Encodes cloth (3+1ch) and agnostic body (3ch) into feature pyramids,
    computes multi-scale cross-correlations, then decodes to a dense
    flow field at full resolution.

    Outputs:
        warped_cloth (B, 3, H, W)  — grid-sampled cloth
        warped_mask  (B, 1, H, W)  — grid-sampled cloth mask
        flow         (B, 2, H, W)  — for regularization loss
    """

    def __init__(self, ngf: int = 64):
        super().__init__()

        # Encoders
        self.cloth_enc  = _FeaturePyramid(in_c=4,  ngf=ngf)   # cloth(3)+mask(1)
        self.body_enc   = _FeaturePyramid(in_c=3,  ngf=ngf)   # agnostic(3)

        # Cross-correlation at each scale
        self.corr4 = _CrossCorrelation(ngf * 8, ngf * 8)
        self.corr3 = _CrossCorrelation(ngf * 4, ngf * 4)
        self.corr2 = _CrossCorrelation(ngf * 2, ngf * 2)
        self.corr1 = _CrossCorrelation(ngf,     ngf)

        # Flow decoder (coarse to fine)
        # Starts from corr4 (H/16), upsamples through corr levels
        self.flow_up4 = UpBlock(ngf * 8, ngf * 4, ngf * 4)   # H/8
        self.flow_up3 = UpBlock(ngf * 4, ngf * 2, ngf * 2)   # H/4
        self.flow_up2 = UpBlock(ngf * 2, ngf,     ngf)        # H/2
        self.flow_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ngf, ngf // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf // 2),
            nn.ReLU(inplace=True),
        )
        self.flow_out = nn.Conv2d(ngf // 2, 2, 3, 1, 1)

    def forward(self, cloth: torch.Tensor, cloth_mask: torch.Tensor,
                agnostic: torch.Tensor):
        """
        Args:
            cloth       (B, 3, H, W)
            cloth_mask  (B, 1, H, W)
            agnostic    (B, 3, H, W)

        Returns:
            warped_cloth (B, 3, H, W)
            warped_mask  (B, 1, H, W)
            flow         (B, 2, H, W)  raw flow for regularization
        """
        cloth_in = torch.cat([cloth, cloth_mask], dim=1)

        # Encode
        cf1, cf2, cf3, cf4 = self.cloth_enc(cloth_in)
        bf1, bf2, bf3, bf4 = self.body_enc(agnostic)

        # Multi-scale correlation
        c4 = self.corr4(cf4, bf4)   # H/16
        c3 = self.corr3(cf3, bf3)   # H/8
        c2 = self.corr2(cf2, bf2)   # H/4
        c1 = self.corr1(cf1, bf1)   # H/2

        # Decode to dense flow (coarse→fine)
        d4 = self.flow_up4(c4, c3)   # H/8
        d3 = self.flow_up3(d4, c2)   # H/4
        d2 = self.flow_up2(d3, c1)   # H/2
        d1 = self.flow_up1(d2)        # H
        flow = self.flow_out(d1)       # (B, 2, H, W)

        # Build sampling grid from flow
        B, _, H, W = cloth.shape
        # Create base identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=cloth.device),
            torch.linspace(-1, 1, W, device=cloth.device),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        base_grid = base_grid.expand(B, -1, -1, -1)

        # Flow is normalized offset, permute to (B, H, W, 2)
        flow_grid = flow.permute(0, 2, 3, 1) * 0.1   # scale factor prevents over-warp
        sample_grid = (base_grid + flow_grid).clamp(-1, 1)

        # Warp cloth and mask
        warped_cloth = F.grid_sample(cloth,      sample_grid,
                                     padding_mode="border", align_corners=True)
        warped_mask  = F.grid_sample(cloth_mask, sample_grid,
                                     padding_mode="zeros",  align_corners=True)
        return warped_cloth, warped_mask, flow


# ---------------------------------------------------------------------------
# Content Fusion Network (CFN)
# ---------------------------------------------------------------------------

class ContentFusionNet(nn.Module):
    """
    U-Net synthesis network that fuses warped cloth with the body representation.

    Input  : agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18) = 25ch
    Output : try-on result (3ch, tanh)

    Parser-free: no human parsing map needed at inference — only agnostic +
    cloth + pose are required.
    """

    def __init__(self, in_channels: int = 25, ngf: int = 64):
        super().__init__()

        # Encoder
        self.e1 = DownBlock(in_channels, ngf)        # H/2
        self.e2 = DownBlock(ngf,         ngf * 2)    # H/4
        self.e3 = DownBlock(ngf * 2,     ngf * 4)    # H/8
        self.e4 = DownBlock(ngf * 4,     ngf * 8)    # H/16
        self.e5 = DownBlock(ngf * 8,     ngf * 8)    # H/32

        # Bottleneck with residual
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
        )

        # Decoder
        self.d5 = UpBlock(ngf * 8, ngf * 8, ngf * 8)
        self.d4 = UpBlock(ngf * 8, ngf * 4, ngf * 4)
        self.d3 = UpBlock(ngf * 4, ngf * 2, ngf * 2)
        self.d2 = UpBlock(ngf * 2, ngf,     ngf)

        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(ngf, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        b  = e5 + self.bottleneck(e5)   # residual bottleneck

        d5 = self.d5(b,  e4)
        d4 = self.d4(d5, e3)
        d3 = self.d3(d4, e2)
        d2 = self.d2(d3, e1)

        feat = self.up_final(d2)
        return torch.tanh(self.out(feat))
