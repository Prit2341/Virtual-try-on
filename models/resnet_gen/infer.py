#!/usr/bin/env python3
"""
ResNet generator inference — saves side-by-side visual strips.

Strip columns: person | cloth | agnostic | warped | output

Usage:
  python models/resnet_gen/infer.py --n 8
  python models/resnet_gen/infer.py --n 16 --data dataset/test/tensors
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

from model.warp_model import WarpNet
from model.warp_utils import warp_cloth
from models.resnet_gen.network import ResNetGenerator
from shared.dataset import VITONDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(ckpt_dir: Path) -> tuple:
    warp_net = WarpNet(in_channels=25, ngf=64, flow_scale=0.25).to(DEVICE)
    warp_path = ckpt_dir / "warp_best.pth"
    if warp_path.exists():
        state = torch.load(warp_path, map_location=DEVICE, weights_only=False)
        warp_net.load_state_dict(state["model"])
        print(f"Loaded WarpNet from {warp_path}")
    else:
        print(f"WARNING: {warp_path} not found — using random WarpNet weights")
    warp_net.eval()

    gen = ResNetGenerator(in_channels=25, ngf=64).to(DEVICE)
    gen_path = ckpt_dir / "resnet_gen_best.pth"
    if gen_path.exists():
        state = torch.load(gen_path, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(state["model"])
        print(f"Loaded ResNetGenerator from {gen_path}")
    else:
        print(f"WARNING: {gen_path} not found — using random generator weights")
    gen.eval()

    return warp_net, gen


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    warp_net, gen = load_models(ckpt_dir)

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

        # Stage 1: warp
        warp_inp = torch.cat([ag, pose, cl, cm], dim=1)
        flow     = warp_net(warp_inp)
        warped   = warp_cloth(cl, flow)
        wm       = warp_cloth(cm, flow)

        # Stage 2: generate
        tryon_inp = torch.cat([ag, warped, wm, pose], dim=1)
        output    = gen(tryon_inp)

        # Build side-by-side strip: person | cloth | agnostic | warped | output
        # All images are (B,3,H,W); wm is (B,1,H,W) → expand to 3ch for display
        strip = torch.cat([person, cl, ag, warped, output], dim=0)  # (5B, 3, H, W)

        # Rearrange so each row is one sample: nrow = B means 5 columns per sample
        save_path = save_dir / "results_strip.jpg"
        save_image(strip, save_path, nrow=len(person), normalize=True, value_range=(-1, 1))
        print(f"Saved strip to {save_path}")
        break  # single batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet generator inference")
    p.add_argument("--n",       type=int, default=8, help="number of test samples")
    p.add_argument("--data",    default=str(ROOT / "dataset" / "test" / "tensors"))
    p.add_argument("--save",    default=str(ROOT / "results" / "resnet_gen"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "resnet_gen"),
                   dest="ckpt_dir")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
