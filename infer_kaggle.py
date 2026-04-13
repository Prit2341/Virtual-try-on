#!/usr/bin/env python3
"""
Inference using the Kaggle-trained GMM + TOM checkpoints.
Exactly mirrors the Kaggle notebook architecture.

Usage:
  python infer_kaggle.py --n 8
  python infer_kaggle.py --n 16 --ckpt-dir results/checkpoints --save results/kaggle_infer
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT   = Path(__file__).resolve().parent


# ════════════════════════════════════════════════════════════
# Models — exact copies from the Kaggle notebook
# ════════════════════════════════════════════════════════════

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        bc = base_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, bc,   3, padding=1),
            nn.BatchNorm2d(bc),   nn.ReLU(inplace=True),
            nn.Conv2d(bc,         bc,   3, padding=1),
            nn.BatchNorm2d(bc),   nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(bc,         bc*2, 3, padding=1),
            nn.BatchNorm2d(bc*2), nn.ReLU(inplace=True),
            nn.Conv2d(bc*2,       bc*2, 3, padding=1),
            nn.BatchNorm2d(bc*2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(bc*2,       bc*4, 3, padding=1),
            nn.BatchNorm2d(bc*4), nn.ReLU(inplace=True),
            nn.Conv2d(bc*4,       bc*4, 3, padding=1),
            nn.BatchNorm2d(bc*4), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class CorrelationLayer(nn.Module):
    def __init__(self, max_displacement=4):
        super().__init__()
        self.max_displacement = max_displacement

    def forward(self, f1, f2):
        B, C, H, W = f1.size()
        d = self.max_displacement
        f2_padded = F.pad(f2, [d] * 4)
        corrs = []
        for dy in range(2 * d + 1):
            for dx in range(2 * d + 1):
                shifted = f2_padded[:, :, dy:dy + H, dx:dx + W]
                corrs.append((f1 * shifted).mean(dim=1, keepdim=True))
        return torch.cat(corrs, dim=1)


class TPSRegressor(nn.Module):
    def __init__(self, in_channels, grid_size=5):
        super().__init__()
        num_params = 2 * grid_size * grid_size
        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,         256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_params),
            nn.Tanh()
        )

    def forward(self, x):
        return self.regressor(x)


class TPSWarper(nn.Module):
    def __init__(self, grid_size=5, img_size=(256, 192)):
        super().__init__()
        self.grid_size = grid_size
        self.img_size  = img_size
        xs = torch.linspace(-0.9, 0.9, grid_size)
        ys = torch.linspace(-0.9, 0.9, grid_size)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('source_pts',
                             torch.stack([gx.flatten(), gy.flatten()], dim=-1))

    def forward(self, cloth, theta):
        B = cloth.size(0)
        N = self.grid_size ** 2
        offsets    = theta.view(B, N, 2) * 0.3
        target_pts = self.source_pts.unsqueeze(0) + offsets
        H, W       = self.img_size
        grid       = self._tps_grid(target_pts, B, H, W)
        return F.grid_sample(cloth, grid, align_corners=True, padding_mode='border')

    def _tps_grid(self, target_pts, B, H, W):
        device = target_pts.device
        gy = torch.linspace(-1, 1, H, device=device)
        gx = torch.linspace(-1, 1, W, device=device)
        my, mx = torch.meshgrid(gy, gx, indexing='ij')
        grid   = torch.stack([mx, my], dim=-1)
        grid   = grid.unsqueeze(0).expand(B, -1, -1, -1).clone()
        src    = self.source_pts
        for i in range(src.size(0)):
            sx, sy = src[i, 0], src[i, 1]
            dx   = target_pts[:, i, 0] - sx
            dy   = target_pts[:, i, 1] - sy
            dist = (mx - sx) ** 2 + (my - sy) ** 2
            w    = torch.exp(-dist * 10).unsqueeze(0)
            grid[:, :, :, 0] += dx.view(B, 1, 1) * w
            grid[:, :, :, 1] += dy.view(B, 1, 1) * w
        return grid.clamp(-1, 1)


class GMM(nn.Module):
    def __init__(self, grid_size=5):
        super().__init__()
        self.feat_person = FeatureExtractor(in_channels=41)
        self.feat_cloth  = FeatureExtractor(in_channels=3)
        self.correlation = CorrelationLayer(max_displacement=4)
        corr_ch          = (2 * 4 + 1) ** 2
        self.regressor   = TPSRegressor(in_channels=256 + corr_ch, grid_size=grid_size)
        self.warper      = TPSWarper(grid_size=grid_size, img_size=(256, 192))

    def forward(self, person_repr, cloth):
        fp    = self.feat_person(person_repr)
        fc    = self.feat_cloth(cloth)
        corr  = self.correlation(fp, fc)
        feat  = torch.cat([fp, corr], dim=1)
        theta = self.regressor(feat)
        warped_cloth = self.warper(cloth, theta)
        warped_mask  = self.warper(
            torch.ones(cloth.size(0), 1, *cloth.shape[2:], device=cloth.device), theta)
        return warped_cloth, warped_mask, theta


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x     = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x     = F.pad(x, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class TOM(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = 41 + 3 + 1  # 45

        self.inc        = DoubleConv(in_ch, 64)
        self.down1      = Down(64,   128)
        self.down2      = Down(128,  256)
        self.down3      = Down(256,  512)
        self.down4      = Down(512,  1024)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True)
        )
        self.up1 = Up(1024, 512)
        self.up2 = Up(512,  256)
        self.up3 = Up(256,  128)
        self.up4 = Up(128,  64)

        self.render_head = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Tanh())
        self.mask_head   = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, person_repr, warped_cloth, warped_mask):
        x_in = torch.cat([person_repr, warped_cloth, warped_mask], dim=1)
        x0 = self.inc(x_in)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x4 = self.bottleneck(x4)
        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)
        x  = self.up4(x,  x0)
        rendered  = self.render_head(x)
        comp_mask = self.mask_head(x)
        result    = comp_mask * warped_cloth + (1 - comp_mask) * rendered
        return result, comp_mask


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def build_person_repr(data: dict) -> torch.Tensor:
    """
    Reconstruct the 41-channel person representation used in Kaggle training:
      agnostic (3ch) + parse_onehot (20ch) + pose_map (18ch) = 41ch

    Local tensors store parse_map as a (H,W) label map — convert to one-hot.
    """
    agnostic  = data["agnostic"]        # (3, H, W)
    pose_map  = data["pose_map"]        # (18, H, W)
    parse_lbl = data["parse_map"]       # (H, W) label map

    H, W = agnostic.shape[1], agnostic.shape[2]
    one_hot = torch.zeros(20, H, W, dtype=torch.float32)
    for c in range(20):
        one_hot[c] = (parse_lbl == c).float()

    return torch.cat([agnostic, one_hot, pose_map], dim=0)  # (41, H, W)


@torch.no_grad()
def run(args):
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    gmm = GMM(grid_size=5).to(DEVICE).eval()
    tom = TOM().to(DEVICE).eval()

    gmm_path = ckpt_dir / "gmm_best.pth"
    tom_path = ckpt_dir / "tom_best.pth"

    gmm.load_state_dict(torch.load(gmm_path, map_location=DEVICE, weights_only=False))
    print(f"GMM loaded  : {gmm_path}")

    tom.load_state_dict(torch.load(tom_path, map_location=DEVICE, weights_only=False))
    print(f"TOM loaded  : {tom_path}")

    # Pick samples
    files  = sorted(Path(args.data).glob("*.pt"))
    chosen = random.sample(files, min(args.n, len(files)))
    print(f"\nRunning inference on {len(chosen)} samples -> {save_dir}/\n")

    all_strips = []

    for f in chosen:
        data = torch.load(f, map_location="cpu", weights_only=False)

        person_repr = build_person_repr(data).unsqueeze(0).to(DEVICE)
        cloth       = data["cloth"].unsqueeze(0).to(DEVICE)
        person      = data["person"].unsqueeze(0).to(DEVICE)

        warped_cloth, warped_mask, _ = gmm(person_repr, cloth)
        result, comp_mask            = tom(person_repr, warped_cloth, warped_mask)

        # Save individual strip: person | cloth | warped | mask | result
        mask_vis = comp_mask.expand(-1, 3, -1, -1)
        strip = torch.cat([person, cloth, warped_cloth, mask_vis, result], dim=0)
        save_image(strip, save_dir / f"{f.stem}.jpg",
                   nrow=5, normalize=True, value_range=(-1, 1), padding=2)
        print(f"  {f.stem}")

        all_strips.append(strip)

    # Combined grid
    if all_strips:
        grid_imgs = torch.cat(all_strips, dim=0)
        grid = make_grid(grid_imgs, nrow=5, normalize=True, value_range=(-1, 1), padding=2)
        TF.to_pil_image(grid.clamp(0, 1)).save(save_dir / "results_grid.jpg")
        print(f"\nGrid saved  : {save_dir}/results_grid.jpg")

    print("\nColumns: Person | Cloth | Warped Cloth | Composition Mask | Try-On Result")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--data",     default=str(ROOT / "dataset" / "test" / "tensors"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "results" / "checkpoints"), dest="ckpt_dir")
    p.add_argument("--save",     default=str(ROOT / "results" / "kaggle_infer"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
