#!/usr/bin/env python3
"""
CP-VITON inference — saves side-by-side visual strips.

Strip columns: person | cloth | agnostic | warped_cloth | alpha | output

Usage:
  python models/cp_viton/infer.py --n 8
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model.gmm_model import GMMNet
from models.cp_viton.network import TryOnModule
from shared.dataset import VITONDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(ckpt_dir: Path):
    gmm = GMMNet(in_h=256, in_w=192, grid_size=5, ngf=64).to(DEVICE)
    gmm_path = ckpt_dir / "gmm_best.pth"
    if gmm_path.exists():
        state = torch.load(gmm_path, map_location=DEVICE, weights_only=False)
        gmm.load_state_dict(state["model"])
        print(f"Loaded GMM from {gmm_path}")
    gmm.eval()

    tom = TryOnModule(in_channels=25, ngf=64).to(DEVICE)
    tom_path = ckpt_dir / "tom_best.pth"
    if tom_path.exists():
        state = torch.load(tom_path, map_location=DEVICE, weights_only=False)
        tom.load_state_dict(state["model"])
        print(f"Loaded TOM from {tom_path}")
    tom.eval()

    return gmm, tom


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    gmm, tom = load_models(ckpt_dir)

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

        # GMM warp
        warped_cl, warped_cm, _ = gmm(cl, cm, ag, pose)
        warped_cl = warped_cl * warped_cm

        # TOM synthesis
        tom_inp          = torch.cat([ag, warped_cl, warped_cm, pose], dim=1)
        output, rendered, alpha = tom(tom_inp, warped_cl)

        # Alpha: expand to 3ch for visualization
        alpha_vis = alpha.expand(-1, 3, -1, -1)

        # Strip: person | cloth | warped | alpha | output
        strip     = torch.cat([person, cl, warped_cl, alpha_vis, output], dim=0)
        save_path = save_dir / "results_strip.jpg"
        save_image(strip, save_path, nrow=len(person), normalize=True, value_range=(-1, 1))
        print(f"Saved strip → {save_path}")
        break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CP-VITON inference")
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--data",     default=str(ROOT / "dataset" / "test" / "tensors"))
    p.add_argument("--save",     default=str(ROOT / "results" / "cp_viton"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "cp_viton"),
                   dest="ckpt_dir")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
