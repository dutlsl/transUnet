#!/bin/bash
# Make sure final_v2 directory structure is created
mkdir -p final_v2/Swirski_tables
mkdir -p final_v2/Swirski_overlays

PYTHON_CMD="/home/iulab1/anaconda3/envs/transUnet/bin/python3 -u"

echo "=== [Swirski GPU 1] Swirski FFT-TTA (radius=8, lr=0.01, iterations=3) Inference 시작 (No Preprocess) ==="
CUDA_VISIBLE_DEVICES=1 $PYTHON_CMD exp/exp5_fft_tta.py \
    --radius 8 \
    --lr 0.01 \
    --iterations 3 \
    --ellipse > swirski.log 2>&1

echo "=== Swirski Inference 완료! ==="
