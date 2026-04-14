#!/usr/bin/env python3
"""
VITON-HD inference — saves side-by-side visual strips.

Strip columns: person | cloth | agnostic | warped_cloth | output

Usage:
  python models/viton_hd/infer.py --n 8
  python models/viton_hd/infer.py --n 16 --data dataset/test/tensors
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

from models.viton_hd.network import (
    SegGenerator, GMM, ALIASGenerator,
    make_parse_agnostic_onehot, parse_7_onehot, N_SEG,
)
from shared.dataset import VITONDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(ckpt_dir: Path):
    seg = SegGenerator(input_nc=41, output_nc=N_SEG).to(DEVICE)
    gmm = GMM(input_nc_A=22, input_nc_B=3).to(DEVICE)
    alias = ALIASGenerator(input_nc=24, ngf=64, seg_nc=N_SEG).to(DEVICE)

    for model, name in [(seg, "seg_best.pth"), (gmm, "gmm_best.pth"), (alias, "alias_best.pth")]:
        path = ckpt_dir / name
        if path.exists():
            state = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model"])
            print(f"Loaded {name}")
        else:
            print(f"WARNING: {path} not found — using random weights")
        model.eval()

    return seg, gmm, alias


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    seg, gmm, alias = load_models(ckpt_dir)

    dataset = VITONDataset(args.data, max_samples=args.n)
    loader  = DataLoader(dataset, batch_size=args.n, shuffle=False, num_workers=0)

    for batch in loader:
        ag   = batch["agnostic"].to(DEVICE)
        cl   = batch["cloth"].to(DEVICE)
        cm   = batch["cloth_mask"].to(DEVICE)
        pose = batch["pose_map"].to(DEVICE)
        pers = batch["person"].to(DEVICE)
        pm   = batch["parse_map"].to(DEVICE)

        if cm.dim() == 3:
            cm = cm.unsqueeze(1)

        # Stage 1: Segmentation
        noise    = torch.randn_like(cm)
        parse_ag = make_parse_agnostic_onehot(pm)
        cl_masked = cl * cm
        seg_inp  = torch.cat([cm, cl_masked, parse_ag, pose, noise], dim=1)
        seg_pred = torch.sigmoid(seg(seg_inp))   # logits → probs
        seg_hard = (seg_pred > 0.5).float()
        seg_cloth = seg_hard[:, 2:3]

        # Stage 2: GMM warp
        inp_A = torch.cat([seg_cloth, pose, ag], dim=1)
        theta, grid = gmm(inp_A, cl)
        warped_cl = F.grid_sample(cl, grid, padding_mode='border', align_corners=False)
        warped_cm = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=False)

        # Stage 3: ALIAS synthesis
        misalign  = (seg_cloth - warped_cm).clamp(min=0)
        seg_div   = torch.cat([seg_hard, misalign], dim=1)
        alias_inp = torch.cat([ag, pose, warped_cl], dim=1)
        output    = alias(alias_inp, seg_hard, seg_div, misalign)

        # Strip: person | cloth | agnostic | warped_cloth | output
        strip = torch.cat([pers, cl, ag, warped_cl, output], dim=0)
        save_path = save_dir / "results_strip.jpg"
        save_image(strip, save_path, nrow=len(pers), normalize=True, value_range=(-1, 1))
        print(f"Saved strip to {save_path}")
        break


def parse_args():
    p = argparse.ArgumentParser(description="VITON-HD inference")
    p.add_argument("--n",        type=int, default=8)
    p.add_argument("--data",     default=str(ROOT / "dataset" / "test" / "tensors"))
    p.add_argument("--save",     default=str(ROOT / "results" / "viton_hd"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "viton_hd"),
                   dest="ckpt_dir")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
