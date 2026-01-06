#!/bin/bash

# 1. 设定 Checkpoint 所在的目录
CHECKPOINT_DIR="/home/CONNECT/yfang870/yunhengwang/WorldVLN/log/2026-01-04_23"

# 仅支持8卡
GPUS=("0" "1" "2" "3" "4" "5" "6" "7")
NUM_GPUS=${#GPUS[@]}

# 捕获 Ctrl+C (SIGINT) 信号并终止所有后台进程
trap 'kill $(jobs -p) 2>/dev/null; exit' EXIT

# 2. 获取该目录下所有的 checkpoint 文件，并按版本/数字顺序排序 (sort -V)
# 这样可以确保你是按训练顺序（如 step_5, step_10...）进行评估的
CHECKPOINT_DIRS=$(find $CHECKPOINT_DIR -maxdepth 1 -type d | tail -n +2 | sort -V)  # tail -n +2 排除根目录本身

# 检查是否找到任何子目录
if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "错误: 在 $CHECKPOINT_DIR 中没有找到子目录"
    echo "当前目录内容:"
    ls -la $CHECKPOINT_DIR/
    exit 1
fi

echo "找到以下子目录:"
echo "$CHECKPOINT_DIRS"

# 3. 外层循环：遍历每一个 checkpoint 文件
for CHECKPOINT_PATH in $CHECKPOINT_DIRS; do
    echo "================================================================"
    echo "Starting evaluation for: $CHECKPOINT_PATH"
    echo "================================================================"

    # 内层循环：你原本的 8 卡并行逻辑
    for i in "${!GPUS[@]}"; do
        GPU="${GPUS[$i]}"
        
        # 运行 Python 脚本（每个进程使用不同的 GPU）
        # 注意：这里直接在命令前指定环境变量，比 export 更安全，不会干扰下一次循环
        echo "Running eval_val_unseen on GPU $GPU for $($CHECKPOINT_PATH)"
        export CUDA_VISIBLE_DEVICES=$GPU
        CUDA_VISIBLE_DEVICES=$GPU python eval.py \
            --checkpoint_path "$CHECKPOINT_PATH" \
            --gpu $GPU \
            --gpu_num $NUM_GPUS &
        
        sleep 5
    done

    # 4. 关键点：等待这一个 Checkpoint 的 8 个进程全部跑完，再开始下一个
    wait
    echo "Finished evaluation for: $CHECKPOINT_PATH"
done

echo "All checkpoints in $CHECKPOINT_DIR have been evaluated."