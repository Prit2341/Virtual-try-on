"""
model/networks.py — VITON-HD Network Architectures
====================================================

Components:
  DownBlock          — encoder block (Conv → InstanceNorm → LeakyReLU)
  UpBlock            — decoder block (ConvTranspose + skip → InstanceNorm → ReLU)
  WarpNet            — Stage 1: flow-based cloth warping (U-Net flow predictor)
  TryOnNet           — Stage 2: try-on synthesis (U-Net generator)
  PatchDiscriminator — 70×70 PatchGAN for adversarial training of TryOnNet
  VGGPerceptualLoss  — VGG-19 feature-space perceptual loss (AMP-safe)

Data flow:
  WarpNet:
    cloth(3) + cloth_mask(1) + agnostic(3) + pose(18) → 25 ch
    → flow field (2, H, W) → grid_sample → warped_cloth(3), warped_mask(1)

  TryOnNet:
    agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18) + parse_oh(18) → 43 ch
    → try-on RGB (3, H, W)

Spatial sizes at 512×384 input (6 encoder levels):
  Input : 512 × 384
  down1 : 256 × 192  ← s1  (ngf    ch)
  down2 : 128 × 96   ← s2  (ngf×2  ch)
  down3 :  64 × 48   ← s3  (ngf×4  ch)
  down4 :  32 × 24   ← s4  (ngf×8  ch)
  down5 :  16 × 12   ← s5  (ngf×8  ch)
  down6 :   8 × 6        (ngf×8  ch) — bottleneck
  up1–5 mirror down5–1, ending at 256 × 192
  flow_head / out_head  upsamples 256×192 → 512×384
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from model.config import Config


# ─────────────────────────── BUILDING BLOCKS ──────────────────────────────────

class DownBlock(nn.Module):
    """Encoder block: stride-2 Conv → InstanceNorm → LeakyReLU."""

    def __init__(self, c_in: int, c_out: int, norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(c_in, c_out, 4, stride=2, padding=1, bias=False)]
        if norm:
            layers.append(nn.InstanceNorm2d(c_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: ConvTranspose (upsample x) → cat skip → Conv → InstanceNorm → ReLU [+ Dropout]."""

    def __init__(self, c_in: int, c_skip: int, c_out: int, dropout: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1, bias=False)
        layers = [
            nn.Conv2d(c_out + c_skip, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))


# ─────────────────────────────── WARP NET ─────────────────────────────────────

class WarpNet(nn.Module):
    """
    Stage 1 — Flow-based Cloth Warping Network.

    U-Net that predicts a 2-channel displacement field (dx, dy) in [-1, 1].
    The field is passed to F.grid_sample to spatially warp the cloth and its mask.

    Input  (25 ch): cloth(3) + cloth_mask(1) + agnostic(3) + pose_map(18)
    Output        : warped_cloth (3, H, W)
                    warped_mask  (1, H, W)
                    flow         (2, H, W)  ← for TV-loss computation
    """

    def __init__(self, in_ch: int = Config.WARP_IN_CH, ngf: int = Config.NGF):
        super().__init__()
        # ── Encoder ──────────────────────────────────────────────────────────
        self.down1 = DownBlock(in_ch,  ngf,     norm=False)   # → 256 × 192
        self.down2 = DownBlock(ngf,    ngf * 2)               # → 128 × 96
        self.down3 = DownBlock(ngf*2,  ngf * 4)               # →  64 × 48
        self.down4 = DownBlock(ngf*4,  ngf * 8)               # →  32 × 24
        self.down5 = DownBlock(ngf*8,  ngf * 8)               # →  16 × 12
        self.down6 = DownBlock(ngf*8,  ngf * 8)               # →   8 × 6   (bottleneck)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up1 = UpBlock(ngf*8, ngf*8, ngf*8, dropout=True)  # → 16 × 12
        self.up2 = UpBlock(ngf*8, ngf*8, ngf*8, dropout=True)  # → 32 × 24
        self.up3 = UpBlock(ngf*8, ngf*4, ngf*4)                 # → 64 × 48
        self.up4 = UpBlock(ngf*4, ngf*2, ngf*2)                 # → 128 × 96
        self.up5 = UpBlock(ngf*2, ngf,   ngf)                   # → 256 × 192

        # ── Flow head: upsample 256×192 → 512×384 → 2-ch displacement ────────
        # Input: cat(d, s1) = ngf + ngf = 128 channels at 256×192
        self.flow_head = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, 2, 3, padding=1),
            nn.Tanh(),                  # flow in [-1, 1]
        )

    def forward(
        self,
        cloth:      torch.Tensor,   # (B, 3,  H, W)
        cloth_mask: torch.Tensor,   # (B, 1,  H, W)
        agnostic:   torch.Tensor,   # (B, 3,  H, W)
        pose_map:   torch.Tensor,   # (B, 18, H, W)
    ):
        x = torch.cat([cloth, cloth_mask, agnostic, pose_map], dim=1)  # (B, 25, H, W)

        # ── Encoder ───────────────────────────────────────────────────────────
        s1 = self.down1(x)           # (B, ngf,   256, 192)
        s2 = self.down2(s1)          # (B, ngf×2, 128,  96)
        s3 = self.down3(s2)          # (B, ngf×4,  64,  48)
        s4 = self.down4(s3)          # (B, ngf×8,  32,  24)
        s5 = self.down5(s4)          # (B, ngf×8,  16,  12)
        bn = self.down6(s5)          # (B, ngf×8,   8,   6) — bottleneck

        # ── Decoder ───────────────────────────────────────────────────────────
        d = self.up1(bn, s5)         # (B, ngf×8, 16,  12)
        d = self.up2(d,  s4)         # (B, ngf×8, 32,  24)
        d = self.up3(d,  s3)         # (B, ngf×4, 64,  48)
        d = self.up4(d,  s2)         # (B, ngf×2, 128, 96)
        d = self.up5(d,  s1)         # (B, ngf,   256, 192)

        # ── Flow field ────────────────────────────────────────────────────────
        # cat(d, s1): both are (B, ngf, 256, 192) → (B, 2*ngf, 256, 192)
        # flow_head upsamples → (B, 2, 512, 384)
        flow = self.flow_head(torch.cat([d, s1], dim=1))

        warped_cloth, warped_mask = self._warp(cloth, cloth_mask, flow)
        return warped_cloth, warped_mask, flow

    @staticmethod
    def _warp(
        cloth:      torch.Tensor,   # (B, 3, H, W)
        cloth_mask: torch.Tensor,   # (B, 1, H, W)
        flow:       torch.Tensor,   # (B, 2, H, W)  dx, dy in [-1, 1]
    ):
        """Apply learned flow field to warp cloth and its mask."""
        B, _, H, W = cloth.shape

        # Identity grid in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=cloth.device),
            torch.linspace(-1, 1, W, device=cloth.device),
            indexing="ij",
        )
        identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Add predicted displacement: (B, 2, H, W) → (B, H, W, 2)
        grid = (identity + flow.permute(0, 2, 3, 1)).clamp(-1, 1)

        warped_cloth = F.grid_sample(
            cloth, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        warped_mask = F.grid_sample(
            cloth_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return warped_cloth, warped_mask


# ─────────────────────────────── TRY-ON NET ───────────────────────────────────

class TryOnNet(nn.Module):
    """
    Stage 2 — Try-On Synthesis Network.

    U-Net generator with 6 encoder / 5 decoder levels + output head.
    Synthesizes the final try-on result from agnostic person, warped cloth,
    warped mask, pose map, and one-hot parse map.

    Input  (43 ch): agnostic(3) + warped_cloth(3) + warped_mask(1)
                    + pose_map(18) + parse_one_hot(18)
    Output  (3 ch): try-on RGB in [-1, 1]
    """

    def __init__(self, in_ch: int = Config.TRYON_IN_CH, ngf: int = Config.NGF):
        super().__init__()
        # ── Encoder ──────────────────────────────────────────────────────────
        self.down1 = DownBlock(in_ch, ngf,     norm=False)
        self.down2 = DownBlock(ngf,   ngf * 2)
        self.down3 = DownBlock(ngf*2, ngf * 4)
        self.down4 = DownBlock(ngf*4, ngf * 8)
        self.down5 = DownBlock(ngf*8, ngf * 8)
        self.down6 = DownBlock(ngf*8, ngf * 8)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up1 = UpBlock(ngf*8, ngf*8, ngf*8, dropout=True)
        self.up2 = UpBlock(ngf*8, ngf*8, ngf*8, dropout=True)
        self.up3 = UpBlock(ngf*8, ngf*4, ngf*4)
        self.up4 = UpBlock(ngf*4, ngf*2, ngf*2)
        self.up5 = UpBlock(ngf*2, ngf,   ngf)

        # ── Output head: upsample 256×192 → 512×384 → 3-ch RGB ───────────────
        # Input: cat(d, s1) = ngf + ngf = 128 channels at 256×192
        self.out_head = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, Config.TRYON_OUT_CH, 3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        agnostic:      torch.Tensor,   # (B, 3,  H, W)
        warped_cloth:  torch.Tensor,   # (B, 3,  H, W)
        warped_mask:   torch.Tensor,   # (B, 1,  H, W)
        pose_map:      torch.Tensor,   # (B, 18, H, W)
        parse_one_hot: torch.Tensor,   # (B, 18, H, W)
    ) -> torch.Tensor:
        x = torch.cat(
            [agnostic, warped_cloth, warped_mask, pose_map, parse_one_hot], dim=1
        )   # (B, 43, H, W)

        # ── Encoder ───────────────────────────────────────────────────────────
        s1 = self.down1(x)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        s5 = self.down5(s4)
        bn = self.down6(s5)

        # ── Decoder ───────────────────────────────────────────────────────────
        d = self.up1(bn, s5)
        d = self.up2(d,  s4)
        d = self.up3(d,  s3)
        d = self.up4(d,  s2)
        d = self.up5(d,  s1)    # (B, ngf, 256, 192)

        # cat(d, s1): (B, 2*ngf, 256, 192) → out_head → (B, 3, 512, 384)
        return self.out_head(torch.cat([d, s1], dim=1))


# ─────────────────────────── PATCH DISCRIMINATOR ──────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.
    Classifies overlapping 70×70 patches as real or fake.

    Input: condition(C_cond ch) concatenated with image(3 ch).
    For TryOnNet: C_cond = 43, total input = 46 ch.
    """

    def __init__(self, in_ch: int, ndf: int = Config.NDF):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,   ndf,     4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf,     ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
            # No sigmoid — use with MSELoss (LSGAN) or BCEWithLogitsLoss
        )

    def forward(self, condition: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([condition, image], dim=1))


# ─────────────────────────── VGG PERCEPTUAL LOSS ──────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using frozen VGG-19 relu features.
    Layers: relu1_2, relu2_2, relu3_3, relu4_3.

    AMP-safe: inputs are cast to float32 before entering VGG, so this
    module works correctly under torch.autocast (fp16) contexts.

    Both pred and target should be in [-1, 1]; internally normalised to ImageNet.
    """

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        self.slice1 = vgg[:4]    # relu1_2
        self.slice2 = vgg[4:9]   # relu2_2
        self.slice3 = vgg[9:18]  # relu3_3
        self.slice4 = vgg[18:27] # relu4_3

        self.criterion = nn.L1Loss()

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → ImageNet normalised float32."""
        x = x.float()                      # ensure fp32 (AMP-safe)
        x = (x + 1.0) / 2.0               # → [0, 1]
        mean = self._MEAN.to(x.device)
        std  = self._STD.to(x.device)
        return (x - mean) / std

    def _features(self, x: torch.Tensor):
        x  = self._normalise(x)
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        f4 = self.slice4(f3)
        return f1, f2, f3, f4

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) in [-1, 1]
            target: (B, 3, H, W) in [-1, 1]
        Returns:
            scalar perceptual loss
        """
        with torch.no_grad():
            tf = self._features(target)
        pf = self._features(pred)
        return sum(self.criterion(p, t) for p, t in zip(pf, tf))
