#!/bin/bash
# Make sure final_v2 directory structure is created
mkdir -p final_v2/LPW_tables
mkdir -p final_v2/LPW_overlays

PYTHON_CMD="/home/iulab1/anaconda3/envs/transUnet/bin/python3 -u"

echo "=== [LPW GPU 0] Folders 1 to 11 Inference 시작 (No Preprocess, UE FFT+Zoom) ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD eval_lpw_skipfilter.py \
    --sigma0 1.0 \
    --sigma1 0.5 \
    --ellipse \
    --ue_fft \
    --f_start 1 \
    --f_end 11 \
    --device cuda:0 > lpw_gpu0.log 2>&1 &
LPW_PID0=$!

echo "=== [LPW GPU 1] Folders 12 to 22 Inference 시작 (No Preprocess, UE FFT+Zoom) ==="
CUDA_VISIBLE_DEVICES=1 $PYTHON_CMD eval_lpw_skipfilter.py \
    --sigma0 1.0 \
    --sigma1 0.5 \
    --ellipse \
    --ue_fft \
    --f_start 12 \
    --f_end 22 \
    --device cuda:0 > lpw_gpu1.log 2>&1 &
LPW_PID1=$!

echo "Waiting for LPW GPU 0 (PID: $LPW_PID0) and LPW GPU 1 (PID: $LPW_PID1) to complete..."
wait $LPW_PID0
wait $LPW_PID1
echo "=== LPW Parallel Inference 완료! ==="
