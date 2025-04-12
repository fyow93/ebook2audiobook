#!/bin/bash
# 开启调试模式，跟踪脚本执行
set -x

echo "GPU检测测试脚本开始执行..."

# 检测可用GPU数量
echo "正在检测可用的GPU..."
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

# 获取传入的参数
echo "传入的参数: $@"

# 检查参数中是否包含headless模式
HAS_HEADLESS=false
for arg in "$@"; do
  echo "分析参数: $arg"
  if [[ "$arg" == "--headless" ]]; then
    HAS_HEADLESS=true
    echo "发现headless参数"
    break
  fi
done

echo "headless模式: $HAS_HEADLESS"
echo "GPU数量: $GPU_COUNT"

# 判断是否使用多GPU模式
if [[ $GPU_COUNT -gt 1 ]] && [[ "$HAS_HEADLESS" == "true" ]]; then
  echo "满足多GPU模式条件！将执行多GPU处理"
  
  # 这里实际应该执行multi_gpu.sh脚本
  echo "将执行: ./multi_gpu.sh $@"
else
  echo "不满足多GPU模式条件，将使用标准模式"
  echo "将执行: python app.py $@"
fi

# 打印完整的条件判断
echo "条件判断结果:"
echo "GPU_COUNT > 1: $([[ $GPU_COUNT -gt 1 ]] && echo "true" || echo "false")"
echo "HAS_HEADLESS == true: $([[ "$HAS_HEADLESS" == "true" ]] && echo "true" || echo "false")"
echo "完整条件: $([[ $GPU_COUNT -gt 1 && "$HAS_HEADLESS" == "true" ]] && echo "true" || echo "false")" 