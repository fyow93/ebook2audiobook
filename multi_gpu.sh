#!/bin/bash
# multi_gpu.sh - 启动多GPU处理模式的脚本
# 注释掉调试输出
# set -x

# 创建日志文件，如果已存在则清空
LOG_FILE="output.log"
> $LOG_FILE

# 将所有后续输出重定向到日志文件，同时保留终端输出
exec > >(tee -a "$LOG_FILE") 2>&1

echo "多GPU处理脚本启动..."
echo "所有日志将被保存到 $LOG_FILE"

# 从参数中提取--procs_per_gpu、--balance_workload和--output_dir参数
PROCS_PER_GPU=4  # 默认每个GPU 4个进程
BALANCE_WORKLOAD=1  # 默认开启工作负载均衡
OUTPUT_DIR="output"  # 默认输出目录
APP_ARGS=()

for arg in "$@"; do
    if [[ $arg == --procs_per_gpu=* ]]; then
        PROCS_PER_GPU="${arg#*=}"
    elif [[ $arg == --balance_workload=* ]]; then
        BALANCE_WORKLOAD="${arg#*=}"
    elif [[ $arg == --output_dir=* ]]; then
        OUTPUT_DIR="${arg#*=}"
    elif [[ $arg == --log_file=* ]]; then
        LOG_FILE="${arg#*=}"
        # 如果指定了新的日志文件，重新设置输出重定向
        > $LOG_FILE
        exec > >(tee -a "$LOG_FILE") 2>&1
        echo "日志将被保存到 $LOG_FILE"
    else
        APP_ARGS+=("$arg")
    fi
done

# 验证每GPU进程数不超过8
if [ $PROCS_PER_GPU -gt 8 ]; then
    echo "警告: 每GPU进程数过高 ($PROCS_PER_GPU)，已限制为8"
    PROCS_PER_GPU=8
fi

echo "每个GPU的进程数: $PROCS_PER_GPU"
echo "工作负载均衡: $BALANCE_WORKLOAD"
echo "输出目录: $OUTPUT_DIR"
echo "应用参数: ${APP_ARGS[@]}"

# 检测可用GPU数量 - 使用静默方式
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "检测到 $GPU_COUNT 个GPU，启动多GPU处理..."

# 计算总进程数
TOTAL_PROCS=$((GPU_COUNT * PROCS_PER_GPU))
# 限制总进程数不超过32
if [ $TOTAL_PROCS -gt 32 ]; then
    TOTAL_PROCS=32
    echo "由于硬件限制，最多使用32个进程"
else
    echo "将使用 $PROCS_PER_GPU 进程/GPU，共 $TOTAL_PROCS 个进程"
fi

if [ $GPU_COUNT -le 1 ]; then
    echo "警告: 只检测到 $GPU_COUNT 个GPU，无法使用多GPU模式"
    echo "将使用标准单GPU模式处理"
    python app.py "${APP_ARGS[@]}"
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

# 设置分布式环境 - 无需显示详细设置过程
export CUDA_VISIBLE_DEVICES=$(seq -s "," 0 $(($GPU_COUNT-1)))

# 检查是否安装了必要的包 - 静默执行
echo "正在初始化环境..."
pip install torch==2.1.0 deepspeed==0.12.5 accelerate==0.25.0 >/dev/null 2>&1

# 清理之前的进程组，避免干扰 - 静默执行
if [ -d "/tmp/torch_distributed" ]; then
    rm -rf /tmp/torch_distributed_* >/dev/null 2>&1
fi

# 启动多GPU处理
echo "开始执行分布式处理..."
# 传递每个GPU进程数参数、工作负载均衡参数和输出目录参数
python multirun.py --nproc_per_node=$TOTAL_PROCS --procs_per_gpu=$PROCS_PER_GPU --balance_workload=$BALANCE_WORKLOAD --output_dir=$OUTPUT_DIR "${APP_ARGS[@]}"

# 如果没有提供参数，显示帮助信息
if [ ${#APP_ARGS[@]} -eq 0 ]; then
    echo "使用方法: ./multi_gpu.sh [--procs_per_gpu=N] [--balance_workload=0|1] [--output_dir=DIR] [--log_file=FILE] [app.py参数]"
    echo "选项:"
    echo "  --procs_per_gpu=N     设置每个GPU运行的进程数，默认为4，最大为8"
    echo "  --balance_workload=N  是否开启工作负载均衡(0=关闭,1=开启)，默认1"
    echo "  --output_dir=DIR      设置输出文件的汇总目录，默认为当前目录下的output"
    echo "  --log_file=FILE       设置日志文件路径，默认为output.log"
    echo "例如:"
    echo "  ./multi_gpu.sh --procs_per_gpu=2 --balance_workload=1 --output_dir=my_audiobooks --log_file=conversion.log --headless --ebook path/to/book.epub"
    echo "  ./multi_gpu.sh --headless --ebooks_dir path/to/books/ --output_dir=audiobooks_output --log_file=batch_conversion.log"
    exit 1
fi 