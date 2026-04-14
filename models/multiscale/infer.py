#!/usr/bin/env python3
"""
Multiscale try-on inference — saves side-by-side visual strips.

Strip columns: person | cloth | agnostic | coarse | refined

Usage:
  python models/multiscale/infer.py --n 8
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.multiscale.network import CoarseNet, RefineNet
from shared.dataset import VITONDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COARSE_H, COARSE_W = 128, 96


def load_models(ckpt_dir: Path) -> tuple:
    coarse_net = CoarseNet(ngf=32).to(DEVICE)
    coarse_path = ckpt_dir / "coarse_best.pth"
    if coarse_path.exists():
        state = torch.load(coarse_path, map_location=DEVICE, weights_only=False)
        coarse_net.load_state_dict(state["model"])
        print(f"Loaded CoarseNet from {coarse_path}")
    else:
        print(f"WARNING: {coarse_path} not found")
    coarse_net.eval()

    refine_net = RefineNet(in_channels=28, ngf=64).to(DEVICE)
    refine_path = ckpt_dir / "refine_best.pth"
    if refine_path.exists():
        state = torch.load(refine_path, map_location=DEVICE, weights_only=False)
        refine_net.load_state_dict(state["model"])
        print(f"Loaded RefineNet from {refine_path}")
    else:
        print(f"WARNING: {refine_path} not found")
    refine_net.eval()

    return coarse_net, refine_net


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    coarse_net, refine_net = load_models(ckpt_dir)

    dataset = VITONDataset(args.data, max_samples=args.n)
    loader  = DataLoader(dataset, batch_size=args.n, shuffle=False, num_workers=0)

    for batch in loader:
        ag     = batch["agnostic"].to(DEVICE)
        cl     = batch["cloth"].to(DEVICE)
        cm     = batch["cloth_mask"].to(DEVICE)
        pose   = batch["pose_map"].to(DEVICE)
        person = batch["person"].to(DEVICE)

        if cm.dim() == 3:
            cm = cm.unsqueeze(1)

        def _down(t):
            return F.interpolate(t, size=(COARSE_H, COARSE_W),
                                 mode="bilinear", align_corners=True)

        def _up_like(t, ref):
            return F.interpolate(t, size=ref.shape[2:],
                                 mode="bilinear", align_corners=True)

        # Coarse stage
        ag_d, cl_d, cm_d, pose_d = _down(ag), _down(cl), _down(cm), _down(pose)
        coarse, warped_d, wm_d = coarse_net(ag_d, cl_d, cm_d, pose_d)

        coarse_up   = _up_like(coarse,   person)
        warped_full = _up_like(warped_d, person)
        wm_full     = _up_like(wm_d,     person)

        # Refine stage
        refine_inp = torch.cat([ag, warped_full, wm_full, coarse_up, pose], dim=1)
        refined    = refine_net(refine_inp)

        # Strip: person | cloth | agnostic | coarse | refined
        strip     = torch.cat([person, cl, ag, coarse_up, refined], dim=0)
        save_path = save_dir / "results_strip.jpg"
        save_image(strip, save_path, nrow=len(person), normalize=True, value_range=(-1, 1))
        print(f"Saved strip to {save_path}")
        break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiscale try-on inference")
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--data",     default=str(ROOT / "dataset" / "test" / "tensors"))
    p.add_argument("--save",     default=str(ROOT / "results" / "multiscale"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "multiscale"),
                   dest="ckpt_dir")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
