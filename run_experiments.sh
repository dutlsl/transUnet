#!/bin/bash
PYTHON_BIN="/home/iulab1/anaconda3/envs/transUnet/bin/python"

echo "=== Exp A (Baseline) ==="
$PYTHON_BIN zeroshot_swirski_skipfilter.py --sigma 0.0

echo "=== Exp B (Ellipse Only) ==="
$PYTHON_BIN zeroshot_swirski_skipfilter.py --sigma 0.0 --ellipse

echo "=== Exp C (Skip Filter Only, sigma=1.0) ==="
$PYTHON_BIN zeroshot_swirski_skipfilter.py --sigma 1.0

echo "=== Exp D (Skip Filter + Ellipse) ==="
$PYTHON_BIN zeroshot_swirski_skipfilter.py --sigma 1.0 --ellipse
