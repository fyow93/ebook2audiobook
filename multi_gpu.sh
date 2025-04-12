#!/bin/bash
# multi_gpu.sh - 启动多GPU处理模式的脚本
# 开启调试输出
set -x

echo "多GPU处理脚本启动..."
echo "传入的参数: $@"

# 检测可用GPU数量
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "检测到 $GPU_COUNT 个可用GPU"

if [ $GPU_COUNT -le 1 ]; then
    echo "警告: 只检测到 $GPU_COUNT 个GPU，无法使用多GPU模式"
    echo "将使用标准单GPU模式处理"
    python app.py "$@"
    exit $?
fi

# 确保脚本可执行
chmod +x multirun.py

# 确保multirun.py存在
if [ ! -f "multirun.py" ]; then
    echo "错误: 未找到multirun.py文件，无法启动多GPU处理"
    exit 1
fi

# 清除可能存在的旧环境变量
unset MASTER_ADDR
unset MASTER_PORT
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK

# 设置分布式环境
export CUDA_VISIBLE_DEVICES=$(seq -s "," 0 $(($GPU_COUNT-1)))
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

# 清理之前的进程组，避免干扰
if [ -d "/tmp/torch_distributed" ]; then
    echo "清理旧的分布式文件..."
    rm -rf /tmp/torch_distributed_*
fi

# 启动多GPU处理
echo "使用 $GPU_COUNT 个GPU启动分布式处理..."
python multirun.py --nproc_per_node=$GPU_COUNT $ARGS 