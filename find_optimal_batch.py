#!/usr/bin/env python3
"""
GPU Batch Size Finder
=====================
Tests each model at increasing batch sizes (forward + backward pass)
to find the maximum batch that fits in VRAM, then recommends an
optimal batch leaving a ~15% safety buffer.

Usage:
    python find_optimal_batch.py
    python find_optimal_batch.py --models single_stage resnet_gen
    python find_optimal_batch.py --max-batch 128 --step 8
"""

import sys, gc, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W   = 256, 192

# ── helpers ─────────────────────────────────────────────────────────────────

def mb(n_bytes):
    return n_bytes / 1024 ** 2

def vram_allocated():
    return torch.cuda.memory_allocated() if DEVICE == "cuda" else 0

def vram_reserved():
    """Reserved by PyTorch caching allocator — actual pages held on GPU.
    When this exceeds total VRAM the driver starts spilling to system RAM."""
    return torch.cuda.memory_reserved() if DEVICE == "cuda" else 0

def vram_peak_reserved():
    return torch.cuda.max_memory_reserved() if DEVICE == "cuda" else 0

def vram_total():
    return torch.cuda.get_device_properties(0).total_memory if DEVICE == "cuda" else 0

def cpu_ram_mb():
    if _PSUTIL:
        return psutil.Process().memory_info().rss / 1024 ** 2
    return 0.0

def clear():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# keep old name used in a few places
def vram_used():
    return vram_allocated()

def fake_batch(b):
    """Return a dict of random input tensors for one batch."""
    return {
        "agnostic":   torch.randn(b, 3,  H, W, device=DEVICE),
        "cloth":      torch.randn(b, 3,  H, W, device=DEVICE),
        "cloth_mask": torch.randn(b, 1,  H, W, device=DEVICE).clamp(0, 1),
        "pose_map":   torch.randn(b, 18, H, W, device=DEVICE),
        "person":     torch.randn(b, 3,  H, W, device=DEVICE),
    }

# ── model builders ───────────────────────────────────────────────────────────

def build_baseline():
    from model.warp_model  import WarpNet
    from model.tryon_model import TryOnNet
    from model.warp_utils  import warp_cloth
    warp  = WarpNet().to(DEVICE)
    tryon = TryOnNet().to(DEVICE)
    params = list(warp.parameters()) + list(tryon.parameters())
    opt = torch.optim.Adam(params, lr=2e-4)
    def forward(d):
        ag, cl, cm, pose, per = d["agnostic"], d["cloth"], d["cloth_mask"], d["pose_map"], d["person"]
        flow = warp(torch.cat([ag, pose, cl, cm], 1))
        warped = warp_cloth(cl, flow)
        wm     = warp_cloth(cm, flow)
        fake   = tryon(torch.cat([ag, warped, wm, pose], 1))
        loss   = F.l1_loss(fake, per) + F.l1_loss(warped * wm, per * wm)
        return loss
    return forward, opt

def build_v2():
    from model.gmm_model      import GMMNet
    from model.tryon_model_v2 import TryOnNetV2
    gmm   = GMMNet().to(DEVICE)
    tryon = TryOnNetV2().to(DEVICE)
    params = list(gmm.parameters()) + list(tryon.parameters())
    opt = torch.optim.Adam(params, lr=2e-4)
    # GMMNet uses torch.linalg.lstsq (TPS solver) which doesn't support fp16.
    # Run entire v2 forward in fp32 regardless of AMP setting.
    def forward(d):
        ag, cl, cm, pose, per = d["agnostic"], d["cloth"], d["cloth_mask"], d["pose_map"], d["person"]
        with torch.amp.autocast("cuda", enabled=False):
            ag2, cl2, cm2, pose2, per2 = ag.float(), cl.float(), cm.float(), pose.float(), per.float()
            warped_cl, warped_m, _ = gmm(cl2, cm2, ag2, pose2)
            out, rendered, alpha   = tryon(torch.cat([ag2, warped_cl, warped_m, pose2], 1), warped_cloth=warped_cl)
            loss = F.l1_loss(out, per2) + 0.1 * alpha.mean()
        return loss
    return forward, opt

def build_resnet_gen():
    from model.warp_model         import WarpNet
    from model.warp_utils         import warp_cloth
    from models.resnet_gen.network import ResNetGenerator
    warp  = WarpNet().to(DEVICE)
    gen   = ResNetGenerator().to(DEVICE)
    params = list(warp.parameters()) + list(gen.parameters())
    opt = torch.optim.Adam(params, lr=2e-4)
    def forward(d):
        ag, cl, cm, pose, per = d["agnostic"], d["cloth"], d["cloth_mask"], d["pose_map"], d["person"]
        flow   = warp(torch.cat([ag, pose, cl, cm], 1))
        warped = warp_cloth(cl, flow)
        wm     = warp_cloth(cm, flow)
        fake   = gen(torch.cat([ag, warped, wm, pose], 1))
        loss   = F.l1_loss(fake, per)
        return loss
    return forward, opt

def build_attention_unet():
    from models.attention_unet.network import AttentionWarpNet, AttentionTryOnNet
    from model.warp_utils import warp_cloth
    warp  = AttentionWarpNet().to(DEVICE)
    tryon = AttentionTryOnNet().to(DEVICE)
    params = list(warp.parameters()) + list(tryon.parameters())
    opt = torch.optim.Adam(params, lr=2e-4)
    def forward(d):
        ag, cl, cm, pose, per = d["agnostic"], d["cloth"], d["cloth_mask"], d["pose_map"], d["person"]
        flow   = warp(torch.cat([ag, pose, cl, cm], 1))
        warped = warp_cloth(cl, flow)
        wm     = warp_cloth(cm, flow)
        fake   = tryon(torch.cat([ag, warped, wm, pose], 1))
        loss   = F.l1_loss(fake, per)
        return loss
    return forward, opt

def build_single_stage():
    from models.single_stage.network import SingleStageTryOn
    net = SingleStageTryOn().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    def forward(d):
        ag, cl, cm, pose, per = d["agnostic"], d["cloth"], d["cloth_mask"], d["pose_map"], d["person"]
        fake = net(torch.cat([ag, cl, cm, pose], 1))
        loss = F.l1_loss(fake, per)
        return loss
    return forward, opt

def build_spade():
    from model.warp_model      import WarpNet
    from model.warp_utils      import warp_cloth
    from models.spade.network  import SPADETryOnNet
    warp  = WarpNet().to(DEVICE)
    tryon = SPADETryOnNet().to(DEVICE)
    params = list(warp.parameters()) + list(tryon.parameters())
    opt = torch.optim.Adam(params, lr=2e-4)
    def forward(d):
        ag, cl, cm, pose, per = d["agnostic"], d["cloth"], d["cloth_mask"], d["pose_map"], d["person"]
        flow   = warp(torch.cat([ag, pose, cl, cm], 1))
        warped = warp_cloth(cl, flow)
        wm     = warp_cloth(cm, flow)
        fake   = tryon(torch.cat([ag, warped, wm, pose], 1), pose)
        loss   = F.l1_loss(fake, per)
        return loss
    return forward, opt

def build_multiscale():
    from models.multiscale.network import CoarseNet, RefineNet
    from model.warp_utils          import warp_cloth
    coarse = CoarseNet().to(DEVICE)
    refine = RefineNet().to(DEVICE)
    params = list(coarse.parameters()) + list(refine.parameters())
    opt = torch.optim.Adam(params, lr=2e-4)
    def forward(d):
        ag, cl, cm, pose, per = d["agnostic"], d["cloth"], d["cloth_mask"], d["pose_map"], d["person"]
        # Coarse at half-res
        ag_d   = F.interpolate(ag,   scale_factor=0.5, mode="bilinear", align_corners=True)
        cl_d   = F.interpolate(cl,   scale_factor=0.5, mode="bilinear", align_corners=True)
        cm_d   = F.interpolate(cm,   scale_factor=0.5, mode="bilinear", align_corners=True)
        pose_d = F.interpolate(pose, scale_factor=0.5, mode="bilinear", align_corners=True)
        coarse_out, w_d, wm_d = coarse(ag_d, cl_d, cm_d, pose_d)
        coarse_up = F.interpolate(coarse_out, size=(H, W), mode="bilinear", align_corners=True)
        w_up  = F.interpolate(w_d,  size=(H, W), mode="bilinear", align_corners=True)
        wm_up = F.interpolate(wm_d, size=(H, W), mode="bilinear", align_corners=True)
        rf_in = torch.cat([ag, w_up, wm_up, coarse_up, pose], 1)
        fine  = refine(rf_in)
        loss  = F.l1_loss(fine, per) + 0.5 * F.l1_loss(coarse_up, per)
        return loss
    return forward, opt

MODELS = {
    "baseline":      build_baseline,
    "v2":            build_v2,
    "resnet_gen":    build_resnet_gen,
    "attention_unet": build_attention_unet,
    "single_stage":  build_single_stage,
    "spade":         build_spade,
    "multiscale":    build_multiscale,
}

# ── benchmark ────────────────────────────────────────────────────────────────

def benchmark_model(name, builder, batch_sizes, use_amp):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    results   = []
    prev_ms   = None   # ms/sample of previous batch (spillage detection)
    total_mb  = mb(vram_total())
    # Hard limit: never attempt a batch if reserved VRAM is already >75%
    # after the previous run (caching allocator retains pages between iters)
    VRAM_HARD_LIMIT = 0.75

    for bs in batch_sizes:
        # Pre-check: if VRAM is already heavily reserved from previous iter,
        # the next larger batch will definitely spill — stop now
        if DEVICE == "cuda":
            already_reserved_pct = torch.cuda.memory_reserved() / vram_total()
            if already_reserved_pct > VRAM_HARD_LIMIT:
                print(f"  batch={bs:4d}  SKIPPED — VRAM already {already_reserved_pct*100:.1f}% reserved, would spill")
                break

        clear()
        try:
            forward_fn, opt = builder()

            data = fake_batch(bs)
            clear()
            forward_fn, opt = builder()
            clear()
            data = fake_batch(bs)

            # Warmup pass
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = forward_fn(data)
            loss.backward()
            opt.zero_grad()

            # After warmup, check reserved VRAM before committing to timed run
            if DEVICE == "cuda":
                post_warmup_pct = torch.cuda.memory_reserved() / vram_total()
                if post_warmup_pct > VRAM_HARD_LIMIT:
                    print(f"  batch={bs:4d}  STOPPED after warmup — reserved {post_warmup_pct*100:.1f}% VRAM (would spill)")
                    del forward_fn, opt, data, loss
                    clear()
                    break

            clear()
            data = fake_batch(bs)

            # Record CPU RAM before timed run
            cpu_before = cpu_ram_mb()

            # Timed run
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = forward_fn(data)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # VRAM: use reserved (actual pages held) not just allocated
            peak_res  = mb(vram_peak_reserved())
            peak_alloc = mb(torch.cuda.max_memory_allocated())
            cpu_after = cpu_ram_mb()
            cpu_delta = cpu_after - cpu_before

            pct_res   = peak_res  / total_mb * 100
            ms_per    = elapsed_ms / bs

            # ── spillage detection ──────────────────────────────────────────
            # 1. Reserved VRAM > 90% of total → getting close to spillage
            # 2. ms/sample is >40% slower than previous batch (proportional
            #    to batch size) — sudden slowdown = driver paging to CPU RAM
            # 3. CPU RAM grew by >200 MB during the GPU pass
            spill_flags = []
            if peak_res > total_mb * 0.75:
                spill_flags.append("VRAM>75%")
            if prev_ms is not None and ms_per > prev_ms * 1.4:
                spill_flags.append(f"SLOW({ms_per:.1f}vs{prev_ms:.1f}ms)")
            if _PSUTIL and cpu_delta > 200:
                spill_flags.append(f"CPU+{cpu_delta:.0f}MB")

            if spill_flags:
                status = "SPILL? " + " ".join(spill_flags)
            elif pct_res > 85:
                status = "HIGH (>85%)"
            else:
                status = "OK"

            results.append((bs, peak_res, peak_alloc, ms_per, pct_res, status))
            cpu_info = f"  cpu+{cpu_delta:.0f}MB" if _PSUTIL else ""
            print(f"  batch={bs:4d}  reserved={peak_res:7.1f}MB ({pct_res:4.1f}%)  "
                  f"alloc={peak_alloc:7.1f}MB  {ms_per:6.2f}ms/sample{cpu_info}  {status}")

            prev_ms = ms_per

            # Stop if spillage detected — next batch will definitely be worse
            if spill_flags:
                print(f"         ^ spillage detected — stopping here")
                del forward_fn, opt, data, loss
                clear()
                break

            del forward_fn, opt, data, loss
            clear()

        except torch.cuda.OutOfMemoryError:
            results.append((bs, None, None, None, None, "OOM"))
            print(f"  batch={bs:4d}  OUT OF MEMORY — stopping")
            clear()
            break
        except Exception as e:
            results.append((bs, None, None, None, None, f"ERROR: {e}"))
            print(f"  batch={bs:4d}  ERROR: {e}")
            clear()
            break

    # Find recommendation — last batch with status OK and reserved <85%
    safe = [(bs, res, ms, pct) for bs, res, alloc, ms, pct, st in results
            if st == "OK" and pct is not None and pct <= 85]
    ok   = [(bs, res, ms, pct) for bs, res, alloc, ms, pct, st in results
            if pct is not None and st not in ("OOM",) and not st.startswith("ERROR")]

    if safe:
        rec_bs, rec_peak, rec_ms, rec_pct = safe[-1]
        print(f"\n  RECOMMENDED batch={rec_bs}  ({rec_peak:.0f} MB reserved / {rec_pct:.1f}% VRAM)  {rec_ms:.2f} ms/sample")
    elif ok:
        rec_bs, rec_peak, rec_ms, rec_pct = ok[-1]
        print(f"\n  MAX (spillage risk) batch={rec_bs}  ({rec_peak:.0f} MB / {rec_pct:.1f}% VRAM)  {rec_ms:.2f} ms/sample")
    else:
        print(f"\n  No successful batch found.")

    return results

# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Find optimal training batch size per model")
    p.add_argument("--models",    nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()), help="Models to test")
    p.add_argument("--max-batch", type=int, default=64,   help="Max batch size to try")
    p.add_argument("--start",     type=int, default=8,    help="Starting batch size")
    p.add_argument("--step",      type=int, default=8,    help="Batch size increment")
    p.add_argument("--no-amp",    action="store_true",    help="Disable AMP (fp16)")
    args = p.parse_args()

    if DEVICE != "cuda":
        print("No CUDA GPU found — exiting.")
        return

    use_amp = not args.no_amp
    gpu     = torch.cuda.get_device_name(0)
    total   = mb(vram_total())
    batch_sizes = list(range(args.start, args.max_batch + 1, args.step))
    if args.start not in batch_sizes:
        batch_sizes = [args.start] + batch_sizes

    print(f"\nGPU        : {gpu}")
    print(f"Total VRAM : {total:.0f} MB  ({total/1024:.2f} GB)")
    print(f"AMP (fp16) : {'enabled' if use_amp else 'disabled'}")
    print(f"Image size : {H}×{W}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Models     : {args.models}")

    all_results = {}
    for name in args.models:
        all_results[name] = benchmark_model(name, MODELS[name], batch_sizes, use_amp)

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY — Recommended batch sizes")
    print(f"{'='*70}")
    print(f"  {'Model':<18} {'Rec. Batch':>10}  {'Peak VRAM':>10}  {'VRAM %':>7}  {'ms/sample':>10}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*10}")

    for name, results in all_results.items():
        safe = [(bs, res, ms, pct) for bs, res, alloc, ms, pct, st in results
                if st == "OK" and pct is not None and pct <= 85]
        ok   = [(bs, res, ms, pct) for bs, res, alloc, ms, pct, st in results
                if pct is not None and st not in ("OOM",) and not st.startswith("ERROR")]
        if safe:
            bs, peak, ms, pct = safe[-1]
        elif ok:
            bs, peak, ms, pct = ok[-1]
        else:
            print(f"  {name:<18}  {'N/A':>10}  {'N/A':>10}  {'N/A':>7}  {'N/A':>10}")
            continue
        print(f"  {name:<18}  {bs:>10}  {peak:>8.0f}MB  {pct:>6.1f}%  {ms:>10.2f}")

    print(f"\n  Current batch=8 uses ~2300 MB. With AMP enabled, you can go much higher.")
    print(f"  Larger batches = more stable gradients + faster epoch times.\n")

if __name__ == "__main__":
    main()
