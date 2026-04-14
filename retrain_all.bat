@echo off
REM ========================================================================
REM  RETRAIN ALL 7 MODELS — with all bug fixes applied
REM  Run from: c:\Virtual_try_on\
REM  Expected time: ~20-30 hours total on RTX 4000 Ada 20GB
REM ========================================================================

echo ================================================================
echo  VIRTUAL TRY-ON — FULL RETRAIN (ALL FIXES APPLIED)
echo  Started: %date% %time%
echo ================================================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM ── Model 1: Baseline (WarpNet + TryOnNet) ──────────────────────────
echo.
echo [1/7] BASELINE — WarpNet + TryOnNet U-Net
echo ================================================================
python train.py --stage both --batch 128 --epochs 100
if %errorlevel% neq 0 echo BASELINE FAILED! && goto model2
echo BASELINE COMPLETE

:model2
REM ── Model 2: V2 (GMM TPS + Composition TryOnNet) ────────────────────
echo.
echo [2/7] V2 — GMM TPS warp + Composition TryOnNet (FIXED cloth labels)
echo ================================================================
python train_v2.py --stage both --batch 64 --epochs 100
if %errorlevel% neq 0 echo V2 FAILED! && goto model3
echo V2 COMPLETE

:model3
REM ── Model 3: ResNet Generator ────────────────────────────────────────
echo.
echo [3/7] RESNET GEN — WarpNet + ResNet9 Generator (FIXED warp loss)
echo ================================================================
python models/resnet_gen/train.py --stage both --batch 64
if %errorlevel% neq 0 echo RESNET_GEN FAILED! && goto model4
echo RESNET_GEN COMPLETE

:model4
REM ── Model 4: Attention U-Net ─────────────────────────────────────────
echo.
echo [4/7] ATTENTION U-NET — Self-attention at bottleneck (FIXED warp loss)
echo ================================================================
python models/attention_unet/train.py --stage both --batch 128
if %errorlevel% neq 0 echo ATTENTION_UNET FAILED! && goto model5
echo ATTENTION_UNET COMPLETE

:model5
REM ── Model 5: Single Stage ────────────────────────────────────────────
echo.
echo [5/7] SINGLE STAGE — 5-level U-Net, no explicit warping
echo ================================================================
python models/single_stage/train.py --batch 192
if %errorlevel% neq 0 echo SINGLE_STAGE FAILED! && goto model6
echo SINGLE_STAGE COMPLETE

:model6
REM ── Model 6: SPADE ──────────────────────────────────────────────────
echo.
echo [6/7] SPADE — Spatially adaptive normalisation (FIXED warp loss)
echo ================================================================
python models/spade/train.py --stage both --batch 80
if %errorlevel% neq 0 echo SPADE FAILED! && goto model7
echo SPADE COMPLETE

:model7
REM ── Model 7: Multiscale ─────────────────────────────────────────────
echo.
echo [7/7] MULTISCALE — CoarseNet 128px + RefineNet 256px
echo ================================================================
python models/multiscale/train.py --stage both --batch 192
if %errorlevel% neq 0 echo MULTISCALE FAILED! && goto done
echo MULTISCALE COMPLETE

:done
echo.
echo ================================================================
echo  ALL TRAINING COMPLETE — %date% %time%
echo ================================================================
echo.
echo  Next: Run comparison
echo    python compare_all.py --n 16
echo.
pause
