#!/usr/bin/env python3
"""
SPADE try-on inference — saves side-by-side visual strips.

Strip columns: person | cloth | agnostic | warped | output

Usage:
  python models/spade/infer.py --n 8
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model.warp_model import WarpNet
from model.warp_utils import warp_cloth
from models.spade.network import SPADETryOnNet
from shared.dataset import VITONDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(ckpt_dir: Path) -> tuple:
    warp_net = WarpNet(in_channels=25, ngf=64, flow_scale=0.5).to(DEVICE)
    warp_path = ckpt_dir / "warp_best.pth"
    if warp_path.exists():
        state = torch.load(warp_path, map_location=DEVICE, weights_only=False)
        warp_net.load_state_dict(state["model"])
        print(f"Loaded WarpNet from {warp_path}")
    else:
        print(f"WARNING: {warp_path} not found")
    warp_net.eval()

    spade_net = SPADETryOnNet(in_channels=25, ngf=64, label_nc=18).to(DEVICE)
    spade_path = ckpt_dir / "tryon_best.pth"
    if spade_path.exists():
        state = torch.load(spade_path, map_location=DEVICE, weights_only=False)
        spade_net.load_state_dict(state["model"])
        print(f"Loaded SPADETryOnNet from {spade_path}")
    else:
        print(f"WARNING: {spade_path} not found")
    spade_net.eval()

    return warp_net, spade_net


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    warp_net, spade_net = load_models(ckpt_dir)

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

        warp_inp  = torch.cat([ag, pose, cl, cm], dim=1)
        flow      = warp_net(warp_inp)
        warped    = warp_cloth(cl, flow)
        wm        = warp_cloth(cm, flow)

        tryon_inp = torch.cat([ag, warped, wm, pose], dim=1)
        output    = spade_net(tryon_inp, pose)

        # Strip: person | cloth | agnostic | warped | output
        strip     = torch.cat([person, cl, ag, warped, output], dim=0)
        save_path = save_dir / "results_strip.jpg"
        save_image(strip, save_path, nrow=len(person), normalize=True, value_range=(-1, 1))
        print(f"Saved strip to {save_path}")
        break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPADE try-on inference")
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--data",     default=str(ROOT / "dataset" / "test" / "tensors"))
    p.add_argument("--save",     default=str(ROOT / "results" / "spade"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "spade"),
                   dest="ckpt_dir")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
