#!/bin/bash
# multi_gpu.sh - 启动多GPU处理模式的脚本
# 开启调试输出
set -x

echo "多GPU处理脚本启动..."
echo "传入的参数: $@"

# 确保脚本可执行
chmod +x multirun.py

# 设置环境变量，确保使用所有GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "设置CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 获取参数
ARGS="$@"

# 如果没有提供参数，显示帮助信息
if [ -z "$ARGS" ]; then
    echo "使用方法: ./multi_gpu.sh [app.py参数]"
    echo "例如:"
    echo "  ./multi_gpu.sh --headless --ebook path/to/book.epub"
    echo "  ./multi_gpu.sh --headless --ebooks_dir path/to/books/"
    exit 1
fi

# 检查是否安装了必要的包
echo "安装必要的依赖包..."
pip install torch deepspeed accelerate

# 检测可用GPU数量
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "检测到 $GPU_COUNT 个可用GPU"

# 启动多GPU处理
echo "启动多GPU处理..."
python multirun.py --nproc_per_node=$GPU_COUNT $ARGS 