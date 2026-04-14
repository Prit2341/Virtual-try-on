#!/bin/bash
# ========================================================================
#  RETRAIN ALL 7 MODELS — with all bug fixes applied
#  Run from: c:\Virtual_try_on\
#  Expected time: ~20-30 hours total on RTX 4000 Ada 20GB
# ========================================================================

set -e
echo "================================================================"
echo " VIRTUAL TRY-ON — FULL RETRAIN (ALL FIXES APPLIED)"
echo " Started: $(date)"
echo "================================================================"

# [1/7] Baseline
echo -e "\n[1/7] BASELINE — WarpNet + TryOnNet U-Net"
python train.py --stage both --batch 128 --epochs 100

# [2/7] V2 (FIXED cloth labels)
echo -e "\n[2/7] V2 — GMM TPS warp + Composition TryOnNet"
python train_v2.py --stage both --batch 64 --epochs 100

# [3/7] ResNet Generator (FIXED warp loss)
echo -e "\n[3/7] RESNET GEN — WarpNet + ResNet9 Generator"
python models/resnet_gen/train.py --stage both --batch 64

# [4/7] Attention U-Net (FIXED warp loss)
echo -e "\n[4/7] ATTENTION U-NET — Self-attention at bottleneck"
python models/attention_unet/train.py --stage both --batch 128

# [5/7] Single Stage
echo -e "\n[5/7] SINGLE STAGE — 5-level U-Net, no explicit warping"
python models/single_stage/train.py --batch 192

# [6/7] SPADE (FIXED warp loss)
echo -e "\n[6/7] SPADE — Spatially adaptive normalisation"
python models/spade/train.py --stage both --batch 80

# [7/7] Multiscale
echo -e "\n[7/7] MULTISCALE — CoarseNet 128px + RefineNet 256px"
python models/multiscale/train.py --stage both --batch 192

echo -e "\n================================================================"
echo " ALL TRAINING COMPLETE — $(date)"
echo "================================================================"
echo ""
echo " Next: Run comparison"
echo "   python compare_all.py --n 16"
