#!/usr/bin/env python3
"""
PF-AFN inference — parser-free try-on.

Strip columns: person | cloth | warped_cloth | output

No human parsing map required at inference — only agnostic + cloth + pose.

Usage:
  python models/pfafn/infer.py --n 8
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.pfafn.network import AppearanceFlowNet, ContentFusionNet
from shared.dataset import VITONDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(ckpt_dir: Path):
    afn = AppearanceFlowNet(ngf=64).to(DEVICE)
    afn_path = ckpt_dir / "afn_best.pth"
    if afn_path.exists():
        state = torch.load(afn_path, map_location=DEVICE, weights_only=False)
        afn.load_state_dict(state["model"])
        print(f"Loaded AFN from {afn_path}")
    afn.eval()

    cfn = ContentFusionNet(in_channels=25, ngf=64).to(DEVICE)
    cfn_path = ckpt_dir / "cfn_best.pth"
    if cfn_path.exists():
        state = torch.load(cfn_path, map_location=DEVICE, weights_only=False)
        cfn.load_state_dict(state["model"])
        print(f"Loaded CFN from {cfn_path}")
    cfn.eval()

    return afn, cfn


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    afn, cfn = load_models(ckpt_dir)

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

        # AFN: dense flow warping (parser-free)
        warped_cl, warped_cm, _ = afn(cl, cm, ag)
        warped_cl = warped_cl * warped_cm

        # CFN: synthesis
        cfn_inp = torch.cat([ag, warped_cl, warped_cm, pose], dim=1)
        output  = cfn(cfn_inp)

        # Strip: person | cloth | warped | output
        strip     = torch.cat([person, cl, warped_cl, output], dim=0)
        save_path = save_dir / "results_strip.jpg"
        save_image(strip, save_path, nrow=len(person), normalize=True, value_range=(-1, 1))
        print(f"Saved strip → {save_path}")
        break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PF-AFN inference")
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--data",     default=str(ROOT / "dataset" / "test" / "tensors"))
    p.add_argument("--save",     default=str(ROOT / "results" / "pfafn"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "pfafn"),
                   dest="ckpt_dir")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
