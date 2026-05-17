#!/bin/bash

# 야간 백그라운드 실험 큐(Queue) 스크립트
# GPU별로 폴더를 반절씩 나누어 순차적으로 3개의 실험을 모두 실행합니다.

PYTHON_CMD="/home/iulab1/anaconda3/envs/transUnet/bin/python3 -u"

run_gpu0() {
    echo "=== [GPU 0] Exp 1: Sigma Search 시작 ==="
    CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD exp1_sigma_search.py --f_start 1 --f_end 11 > exp1_gpu0.log 2>&1
    echo "=== [GPU 0] Exp 2: RITnet Preprocess 시작 ==="
    CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD eval_lpw_skipfilter.py --folder 1 --all_folders --preprocess --ellipse > exp2_gpu0.log 2>&1
    echo "=== [GPU 0] Exp 3: Kernel Search 시작 ==="
    CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD exp3_kernel_search.py --f_start 1 --f_end 11 > exp3_gpu0.log 2>&1
    echo "=== [GPU 0] 모든 작업 완료 ==="
}

run_gpu1() {
    echo "=== [GPU 1] Exp 1: Sigma Search 시작 ==="
    CUDA_VISIBLE_DEVICES=1 $PYTHON_CMD exp1_sigma_search.py --f_start 12 --f_end 22 > exp1_gpu1.log 2>&1
    # Exp 2의 eval_lpw_skipfilter는 all_folders로 실행시 전체를 순회하므로, 
    # 폴더를 나누기 위해 스크립트 파라미터를 사용하거나 따로 돌려야 함.
    # 단, eval_lpw_skipfilter.py에는 --f_start / --f_end 가 없고 --folder N 또는 --all_folders만 있음.
    # GPU0에서 --all_folders로 전체를 1시간만에 돌게 냅두면 됨.
    # GPU1에서는 Exp 2를 생략하거나 다른 것을 할 수 있지만, 여기서는 안전하게 Exp 3로 넘어감.
    echo "=== [GPU 1] Exp 3: Kernel Search 시작 ==="
    CUDA_VISIBLE_DEVICES=1 $PYTHON_CMD exp3_kernel_search.py --f_start 12 --f_end 22 > exp3_gpu1.log 2>&1
    echo "=== [GPU 1] 모든 작업 완료 ==="
}

# 백그라운드로 GPU0, GPU1 큐 실행
run_gpu0 &
PID0=$!

run_gpu1 &
PID1=$!

echo "모든 실험이 백그라운드 큐에 등록되었습니다."
echo "GPU 0 PID: $PID0"
echo "GPU 1 PID: $PID1"
