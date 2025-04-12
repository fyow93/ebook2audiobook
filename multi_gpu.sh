#!/bin/bash
# multi_gpu.sh - 启动多GPU处理模式的脚本

# 确保脚本可执行
chmod +x multirun.py

# 设置环境变量，确保使用所有GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

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
pip install torch deepspeed accelerate

# 启动多GPU处理
python multirun.py --nproc_per_node=4 $ARGS 