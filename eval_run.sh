#!/bin/bash

# 检查 checkpoint_path 是否有效
CHECKPOINT_PATH="log/2026-01-04_23/checkpoint_step_5_3381"

# 仅支持8卡
GPUS=("4" "5" "6" "7")

# 运行多个进程，每个进程使用不同的 GPU
# 用于捕获 Ctrl+C (SIGINT) 信号并终止所有后台进程
trap 'kill $(jobs -p)' EXIT

NUM_GPUS=${#GPUS[@]}

for i in "${!GPUS[@]}"; do
    GPU="${GPUS[$i]}"
    
    # 设置 CUDA_VISIBLE_DEVICES 为指定的 GPU
    export CUDA_VISIBLE_DEVICES=$GPU
    
    # 运行 Python 脚本（每个进程使用不同的 GPU）
    echo "Running eval_val_unseen on GPU $GPU"
    
    # 在后台运行每个进程 (并行执行)
    python eval.py --checkpoint_path "$CHECKPOINT_PATH" --gpu $GPU --gpu_num $NUM_GPUS &
    
    # 你可以添加 sleep 让进程间稍微有些间隔（可选）
    sleep 1
done

# 等待所有后台进程完成
wait

echo "Evaluation finished on all GPUs."