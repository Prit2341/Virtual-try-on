#!/usr/bin/env python3
"""
Single-stage try-on inference — saves side-by-side visual strips.

Strip columns: person | cloth | agnostic | output
(No warped column since there is no warping stage.)

Usage:
  python models/single_stage/infer.py --n 8
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.single_stage.network import SingleStageTryOn
from shared.dataset import VITONDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_dir: Path) -> SingleStageTryOn:
    model = SingleStageTryOn(in_channels=25, ngf=64).to(DEVICE)
    best_path = ckpt_dir / "model_best.pth"
    if best_path.exists():
        state = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"Loaded SingleStageTryOn from {best_path}")
    else:
        print(f"WARNING: {best_path} not found — using random weights")
    model.eval()
    return model


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt_dir)

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

        inp    = torch.cat([ag, cl, cm, pose], dim=1)
        output = model(inp)

        # Strip: person | cloth | agnostic | output
        strip     = torch.cat([person, cl, ag, output], dim=0)
        save_path = save_dir / "results_strip.jpg"
        save_image(strip, save_path, nrow=len(person), normalize=True, value_range=(-1, 1))
        print(f"Saved strip to {save_path}")
        break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-stage try-on inference")
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--data",     default=str(ROOT / "dataset" / "test" / "tensors"))
    p.add_argument("--save",     default=str(ROOT / "results" / "single_stage"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "single_stage"),
                   dest="ckpt_dir")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
