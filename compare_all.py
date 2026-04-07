#!/usr/bin/env python3
"""
Compare all trained virtual try-on models on the test set.

Computes L1, SSIM, PSNR for each model that has trained checkpoints.
Outputs:
  - Terminal table
  - results/comparison_summary.csv
  - results/comparison_grid.jpg  (N test samples × all trained models)

Models evaluated:
  baseline   : WarpNet + TryOnNet           (checkpoints/)
  v2         : GMMNet + TryOnNetV2          (checkpoints/v2/)
  resnet_gen : WarpNet + ResNetGenerator    (checkpoints/resnet_gen/)
  attention_unet : AttentionWarpNet + AttentionTryOnNet (checkpoints/attention_unet/)
  single_stage : SingleStageTryOn           (checkpoints/single_stage/)
  spade      : WarpNet + SPADETryOnNet      (checkpoints/spade/)
  multiscale : CoarseNet + RefineNet        (checkpoints/multiscale/)

Usage:
  python compare_all.py
  python compare_all.py --n 8 --split test --batch 16
  python compare_all.py --models baseline resnet_gen spade
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF

from shared.dataset import VITONDataset
from shared.losses import VGGLoss
from shared.metrics import ssim_metric, psnr_metric
from model.warp_model import WarpNet
from model.tryon_model import TryOnNet
from model.warp_utils import warp_cloth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_MODELS = ["baseline", "v2", "resnet_gen", "attention_unet",
              "single_stage", "spade", "multiscale"]


# ---------------------------------------------------------------------------
# Per-model inference helpers
# ---------------------------------------------------------------------------

def _unpack(batch: dict) -> tuple:
    ag     = batch["agnostic"].to(DEVICE)
    cl     = batch["cloth"].to(DEVICE)
    cm     = batch["cloth_mask"].to(DEVICE)
    pose   = batch["pose_map"].to(DEVICE)
    person = batch["person"].to(DEVICE)
    if cm.dim() == 3:
        cm = cm.unsqueeze(1)
    return ag, cl, cm, pose, person


def _load_state(path: Path, model: nn.Module, label: str) -> bool:
    if not path.exists():
        print(f"  [{label}] checkpoint not found: {path}")
        return False
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return True


# ---- baseline ---------------------------------------------------------------

def run_baseline(batch: dict, ckpt_dir: Path, cache: dict) -> torch.Tensor | None:
    if "baseline" not in cache:
        wn = WarpNet(in_channels=25, ngf=64, flow_scale=0.25).to(DEVICE)
        tn = TryOnNet(in_channels=25, ngf=64).to(DEVICE)
        ok_w = _load_state(ckpt_dir / "warp_best.pth",  wn, "baseline/warp")
        ok_t = _load_state(ckpt_dir / "tryon_best.pth", tn, "baseline/tryon")
        if not (ok_w and ok_t):
            return None
        cache["baseline"] = (wn, tn)
    wn, tn = cache["baseline"]

    ag, cl, cm, pose, _ = _unpack(batch)
    with torch.no_grad():
        flow   = wn(torch.cat([ag, pose, cl, cm], 1))
        warped = warp_cloth(cl, flow)
        wm     = warp_cloth(cm, flow)
        out    = tn(torch.cat([ag, warped, wm, pose], 1))
    return out


# ---- v2 (GMM + TryOnNetV2) --------------------------------------------------

def run_v2(batch: dict, ckpt_dir: Path, cache: dict) -> torch.Tensor | None:
    v2_dir = ckpt_dir / "v2"
    if "v2" not in cache:
        from model.gmm_model import GMMNet
        from model.tryon_model_v2 import TryOnNetV2

        gmm = GMMNet().to(DEVICE)
        tn2 = TryOnNetV2(in_channels=25, ngf=64).to(DEVICE)
        ok_g = _load_state(v2_dir / "gmm_best.pth",   gmm, "v2/gmm")
        ok_t = _load_state(v2_dir / "tryon_best.pth", tn2, "v2/tryon")
        if not (ok_g and ok_t):
            return None
        cache["v2"] = (gmm, tn2)
    gmm, tn2 = cache["v2"]

    ag, cl, cm, pose, _ = _unpack(batch)
    with torch.no_grad():
        warped, wm, _ = gmm(cl, cm, ag, pose)
        inp = torch.cat([ag, warped, wm, pose], 1)
        out, _, _ = tn2(inp, warped_cloth=warped)
    return out


# ---- resnet_gen -------------------------------------------------------------

def run_resnet_gen(batch: dict, ckpt_dir: Path, cache: dict) -> torch.Tensor | None:
    rg_dir = ckpt_dir / "resnet_gen"
    if "resnet_gen" not in cache:
        from models.resnet_gen.network import ResNetGenerator

        wn  = WarpNet(in_channels=25, ngf=64, flow_scale=0.25).to(DEVICE)
        gen = ResNetGenerator(in_channels=25, ngf=64).to(DEVICE)
        ok_w = _load_state(rg_dir / "warp_best.pth",       wn,  "resnet_gen/warp")
        ok_g = _load_state(rg_dir / "resnet_gen_best.pth", gen, "resnet_gen/gen")
        if not (ok_w and ok_g):
            return None
        cache["resnet_gen"] = (wn, gen)
    wn, gen = cache["resnet_gen"]

    ag, cl, cm, pose, _ = _unpack(batch)
    with torch.no_grad():
        flow   = wn(torch.cat([ag, pose, cl, cm], 1))
        warped = warp_cloth(cl, flow)
        wm     = warp_cloth(cm, flow)
        out    = gen(torch.cat([ag, warped, wm, pose], 1))
    return out


# ---- attention_unet ---------------------------------------------------------

def run_attention_unet(batch: dict, ckpt_dir: Path, cache: dict) -> torch.Tensor | None:
    au_dir = ckpt_dir / "attention_unet"
    if "attention_unet" not in cache:
        from models.attention_unet.network import AttentionWarpNet, AttentionTryOnNet

        wn = AttentionWarpNet(in_channels=25, ngf=64, flow_scale=0.25).to(DEVICE)
        tn = AttentionTryOnNet(in_channels=25, ngf=64).to(DEVICE)
        ok_w = _load_state(au_dir / "warp_best.pth",  wn, "attention_unet/warp")
        ok_t = _load_state(au_dir / "tryon_best.pth", tn, "attention_unet/tryon")
        if not (ok_w and ok_t):
            return None
        cache["attention_unet"] = (wn, tn)
    wn, tn = cache["attention_unet"]

    ag, cl, cm, pose, _ = _unpack(batch)
    with torch.no_grad():
        flow   = wn(torch.cat([ag, pose, cl, cm], 1))
        warped = warp_cloth(cl, flow)
        wm     = warp_cloth(cm, flow)
        out    = tn(torch.cat([ag, warped, wm, pose], 1))
    return out


# ---- single_stage -----------------------------------------------------------

def run_single_stage(batch: dict, ckpt_dir: Path, cache: dict) -> torch.Tensor | None:
    ss_dir = ckpt_dir / "single_stage"
    if "single_stage" not in cache:
        from models.single_stage.network import SingleStageTryOn

        model = SingleStageTryOn(in_channels=25, ngf=64).to(DEVICE)
        ok = _load_state(ss_dir / "model_best.pth", model, "single_stage")
        if not ok:
            return None
        cache["single_stage"] = model
    model = cache["single_stage"]

    ag, cl, cm, pose, _ = _unpack(batch)
    with torch.no_grad():
        out = model(torch.cat([ag, cl, cm, pose], 1))
    return out


# ---- spade ------------------------------------------------------------------

def run_spade(batch: dict, ckpt_dir: Path, cache: dict) -> torch.Tensor | None:
    sp_dir = ckpt_dir / "spade"
    if "spade" not in cache:
        from models.spade.network import SPADETryOnNet

        wn = WarpNet(in_channels=25, ngf=64, flow_scale=0.25).to(DEVICE)
        sn = SPADETryOnNet(in_channels=25, ngf=64, label_nc=18).to(DEVICE)
        ok_w = _load_state(sp_dir / "warp_best.pth",  wn, "spade/warp")
        ok_s = _load_state(sp_dir / "tryon_best.pth", sn, "spade/tryon")
        if not (ok_w and ok_s):
            return None
        cache["spade"] = (wn, sn)
    wn, sn = cache["spade"]

    ag, cl, cm, pose, _ = _unpack(batch)
    with torch.no_grad():
        flow   = wn(torch.cat([ag, pose, cl, cm], 1))
        warped = warp_cloth(cl, flow)
        wm     = warp_cloth(cm, flow)
        inp    = torch.cat([ag, warped, wm, pose], 1)
        out    = sn(inp, pose)
    return out


# ---- multiscale -------------------------------------------------------------

def run_multiscale(batch: dict, ckpt_dir: Path, cache: dict) -> torch.Tensor | None:
    ms_dir = ckpt_dir / "multiscale"
    if "multiscale" not in cache:
        from models.multiscale.network import CoarseNet, RefineNet

        cn = CoarseNet(ngf=32).to(DEVICE)
        rn = RefineNet(in_channels=28, ngf=64).to(DEVICE)
        ok_c = _load_state(ms_dir / "coarse_best.pth", cn, "multiscale/coarse")
        ok_r = _load_state(ms_dir / "refine_best.pth", rn, "multiscale/refine")
        if not (ok_c and ok_r):
            return None
        cache["multiscale"] = (cn, rn)
    cn, rn = cache["multiscale"]

    ag, cl, cm, pose, person = _unpack(batch)
    H, W = person.shape[2], person.shape[3]

    def _down(t):
        return F.interpolate(t, size=(128, 96), mode="bilinear", align_corners=True)

    def _up(t):
        return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=True)

    with torch.no_grad():
        coarse, warped_d, wm_d = cn(_down(ag), _down(cl), _down(cm), _down(pose))
        coarse_up   = _up(coarse)
        warped_full = _up(warped_d)
        wm_full     = _up(wm_d)
        inp = torch.cat([ag, warped_full, wm_full, coarse_up, pose], 1)
        out = rn(inp)
    return out


# ---------------------------------------------------------------------------
# Model dispatch table
# ---------------------------------------------------------------------------

_RUNNERS = {
    "baseline":      run_baseline,
    "v2":            run_v2,
    "resnet_gen":    run_resnet_gen,
    "attention_unet": run_attention_unet,
    "single_stage":  run_single_stage,
    "spade":         run_spade,
    "multiscale":    run_multiscale,
}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    runner,
    loader: DataLoader,
    ckpt_dir: Path,
    cache: dict,
) -> dict | None:
    """Run model on full loader, return averaged metrics. Returns None if not available."""
    total_l1   = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    n_batches  = 0

    l1_fn = nn.L1Loss()

    for batch in loader:
        out = runner(batch, ckpt_dir, cache)
        if out is None:
            return None  # checkpoint not available

        _, _, _, _, person = _unpack(batch)

        total_l1   += l1_fn(out, person).item()
        total_ssim += ssim_metric(out, person).item()
        total_psnr += psnr_metric(out, person).item()
        n_batches  += 1

    if n_batches == 0:
        return None

    return {
        "L1":   total_l1   / n_batches,
        "SSIM": total_ssim / n_batches,
        "PSNR": total_psnr / n_batches,
    }


# ---------------------------------------------------------------------------
# Comparison grid
# ---------------------------------------------------------------------------

def build_grid(
    model_names: list,
    runners: dict,
    samples: list,
    ckpt_dir: Path,
    cache: dict,
    n: int,
) -> torch.Tensor | None:
    """
    Build comparison grid image.
    Rows: each test sample (person | cloth | model1 | model2 | ...)
    """
    # Collect sample batches one by one
    rows = []
    for i, batch in enumerate(samples):
        if i >= n:
            break

        # Person and cloth (single sample)
        ag     = batch["agnostic"].unsqueeze(0).to(DEVICE) if batch["agnostic"].dim() == 3 else batch["agnostic"].to(DEVICE)
        cl     = batch["cloth"].unsqueeze(0).to(DEVICE)    if batch["cloth"].dim() == 3    else batch["cloth"].to(DEVICE)
        person = batch["person"].unsqueeze(0).to(DEVICE)   if batch["person"].dim() == 3   else batch["person"].to(DEVICE)
        cm     = batch["cloth_mask"].to(DEVICE)
        pose   = batch["pose_map"].to(DEVICE)
        if cm.dim() == 2:
            cm = cm.unsqueeze(0).unsqueeze(0)
        elif cm.dim() == 3:
            cm = cm.unsqueeze(0) if cm.shape[0] != person.shape[0] else cm.unsqueeze(1)

        if person.dim() == 3:
            person = person.unsqueeze(0)
        if cl.dim() == 3:
            cl = cl.unsqueeze(0)
        if ag.dim() == 3:
            ag = ag.unsqueeze(0)
        if pose.dim() == 3:
            pose = pose.unsqueeze(0)

        single_batch = {
            "agnostic": ag, "cloth": cl, "cloth_mask": cm,
            "pose_map": pose, "person": person,
        }

        row_imgs = [person[0], cl[0]]  # person, cloth

        for mname in model_names:
            runner = runners[mname]
            out = runner(single_batch, ckpt_dir, cache)
            if out is not None:
                row_imgs.append(out[0])
            else:
                # Blank placeholder
                row_imgs.append(torch.zeros_like(person[0]))

        rows.append(torch.stack(row_imgs, dim=0))  # (2+M, 3, H, W)

    if not rows:
        return None

    # Stack all rows: (N*(2+M), 3, H, W)
    all_imgs = torch.cat(rows, dim=0)
    n_cols   = len(rows[0])
    grid     = make_grid(all_imgs, nrow=n_cols, normalize=True, value_range=(-1, 1), padding=2)
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare all try-on models")
    p.add_argument("--n",      type=int, default=8,   help="samples for comparison grid")
    p.add_argument("--split",  default="test",         help="dataset split directory")
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--data",   default=None,
                   help="path to test tensors (default: dataset/<split>/tensors)")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, help="subset of models to evaluate")
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"), dest="ckpt_dir")
    p.add_argument("--out-dir",  default=str(ROOT / "results"),     dest="out_dir")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data or str(ROOT / "dataset" / args.split / "tensors")

    print(f"Evaluating on: {data_path}")
    print(f"Models: {args.models}")
    print(f"Device: {DEVICE}")

    dataset = VITONDataset(data_path)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                         num_workers=0, drop_last=False)

    cache   = {}  # shared model cache across evaluations
    results = {}  # name -> metrics dict

    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'L1':>8} {'SSIM':>8} {'PSNR (dB)':>10}")
    print("-" * 70)

    for mname in args.models:
        runner  = _RUNNERS[mname]
        metrics = evaluate_model(mname, runner, loader, ckpt_dir, cache)
        if metrics is None:
            print(f"{'  ' + mname:<20} {'N/A':>8} {'N/A':>8} {'N/A':>10}  (no checkpoint)")
        else:
            results[mname] = metrics
            print(f"{'  ' + mname:<20} {metrics['L1']:>8.4f} {metrics['SSIM']:>8.4f} "
                  f"{metrics['PSNR']:>10.2f}")

    print("=" * 70)

    # Save CSV
    csv_path = out_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "L1", "SSIM", "PSNR_dB"])
        for mname, m in results.items():
            writer.writerow([mname, f"{m['L1']:.6f}", f"{m['SSIM']:.6f}", f"{m['PSNR']:.4f}"])
    print(f"\nSaved CSV  : {csv_path}")

    # Build comparison grid
    trained_models = list(results.keys())
    if not trained_models:
        print("No trained models found — skipping grid.")
        return

    # Load individual samples for grid
    sample_dataset = VITONDataset(data_path, max_samples=args.n)
    grid = build_grid(trained_models, _RUNNERS, sample_dataset, ckpt_dir, cache, n=args.n)

    if grid is not None:
        grid_path = out_dir / "comparison_grid.jpg"
        TF.to_pil_image(grid.clamp(0, 1)).save(grid_path)
        print(f"Saved grid : {grid_path}")
        print(f"  Columns  : person | cloth | {' | '.join(trained_models)}")


if __name__ == "__main__":
    main()
