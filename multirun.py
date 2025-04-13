#!/usr/bin/env python
"""
多GPU启动脚本 - 为ebook2audiobook启用分布式处理
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import subprocess
import glob
import psutil
import signal
from pathlib import Path
import tempfile
import shutil
import re

def get_num_gpus():
    """获取可用的GPU数量"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def setup_distributed(rank, world_size):
    """设置分布式环境"""
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # 初始化进程组
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"分布式进程 {rank}/{world_size} 初始化成功")
        
        # 设置当前设备
        torch.cuda.set_device(rank)
        
        return True
    except Exception as e:
        print(f"初始化分布式环境失败: {e}")
        return False

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("已销毁分布式进程组")

def run_worker(rank, world_size, args, gpu_ids, procs_per_gpu):
    """每个工作进程的执行函数"""
    # 设置分布式环境
    success = setup_distributed(rank, world_size)
    if not success:
        print(f"进程 {rank} 无法初始化分布式环境，退出")
        return
    
    # 计算当前进程应该使用的GPU ID
    # 修改GPU分配逻辑，确保进程均匀分布到所有GPU上
    gpu_id = gpu_ids[rank % len(gpu_ids)]
    proc_id_on_gpu = rank // len(gpu_ids)
    
    print(f"工作进程 {rank} 使用 GPU ID: {gpu_id} (GPU {gpu_id} 上的第{proc_id_on_gpu + 1}个进程)")
    
    # 设置DeepSpeed相关环境变量
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__)) + ":" + os.environ.get('PYTHONPATH', '')

    # 准备命令行参数
    cmd_args = args.copy()
    
    # 添加或修改为分布式模式必须的参数
    cmd_args.extend([
        '--gpu_id', str(gpu_id),
        '--use_distributed', 'True',
        '--deepspeed'  # 默认开启DeepSpeed
    ])
    
    # 设置工作分片参数，确保每个进程处理不同的工作
    if '--balance_work' in cmd_args and cmd_args[cmd_args.index('--balance_work') + 1].lower() == 'true':
        # 为每个进程添加工作分片ID和总分片数
        cmd_args.extend([
            '--shard_id', str(rank),
            '--num_shards', str(world_size)
        ])
        print(f"进程 {rank} 被分配处理分片 {rank}/{world_size}")
    
    # 为多本书处理添加特定参数
    if '--parallel_books' in cmd_args and cmd_args[cmd_args.index('--parallel_books') + 1].lower() == 'true':
        # 为每个进程添加书籍分片ID
        cmd_args.extend([
            '--book_shard_id', str(rank),
            '--book_num_shards', str(world_size)
        ])
        print(f"进程 {rank} 将处理第 {rank} 本书 (总共分配 {world_size} 个进程处理多本书)")
    
    # 检查是否已经有--headless参数
    if '--headless' not in cmd_args:
        cmd_args.append('--headless')
        
    # 如果有需要，可以添加DeepSpeed特定的参数，但仅当未指定deepspeed_config时
    if ('--tts_engine' not in cmd_args or ('--tts_engine' in cmd_args and cmd_args[cmd_args.index('--tts_engine') + 1] == 'xtts')) and '--deepspeed_config' not in cmd_args:
        # ====================================================
        # A100 GPU优化说明:
        # 1. 超大batch_size(256)以充分利用40GB显存
        # 2. 降级到ZeRO-2提高速度，减少参数卸载
        # 3. 增加每GPU微批次大小到32
        # 4. 添加激活检查点优化内存使用
        # ====================================================
        
        # 为DeepSpeed创建临时配置文件，与主配置保持一致
        ds_config = {
            "train_batch_size": 256,                 # 训练批次大小，超大批次充分利用A100显存
            "gradient_accumulation_steps": 1,        # 减少梯度累积提高并行度
            "optimizer": {                           # 优化器配置
                "type": "Adam",                      # 使用Adam优化器
                "params": {
                    "lr": 0.0001,                    # 学习率设置
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "fp16": {                                # 半精度浮点数训练
                "enabled": True,                     # 启用FP16，可减少内存使用并加速训练
                "loss_scale": 0,                     # 动态损失缩放
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {                   # ZeRO内存优化
                "stage": 2,                          # 降级到Stage 2提高速度
                "offload_optimizer": {               # 优化器状态卸载配置
                    "device": "cpu",                 # 卸载到CPU
                    "pin_memory": True               # 使用固定内存提高传输效率
                },
                "offload_param": {                   # 参数卸载配置
                    "device": "none"                 # 不卸载参数到CPU以提高速度
                },
                "allgather_partitions": True,        # 在前向/反向传播前收集所有分区
                "allgather_bucket_size": 5e8,        # allgather操作的通信桶大小
                "overlap_comm": True,                # 重叠通信和计算以提高效率
                "reduce_scatter": True,              # 使用reduce-scatter而不是reduce+scatter
                "reduce_bucket_size": 5e8,           # reduce操作的通信桶大小
                "contiguous_gradients": True         # 使用连续的内存缓冲区存储梯度
            },
            "gradient_clipping": 1.0,                # 梯度裁剪阈值，防止梯度爆炸
            "steps_per_print": 10,                   # 打印训练信息的步数间隔
            "train_micro_batch_size_per_gpu": 32,    # 每个GPU的微批次大小，增大以提高利用率
            "wall_clock_breakdown": False,           # 关闭墙钟时间细分
            "activation_checkpointing": {            # 激活检查点优化内存
                "partition_activations": True,       # 分区激活以减少内存使用
                "cpu_checkpointing": False,          # 不使用CPU检查点以提高速度
                "contiguous_memory_optimization": True  # 使用连续内存优化
            },
            "communication_data_type": "fp16"        # 通信数据类型，使用FP16减少通信量
        }
        
        import json
        temp_ds_config = os.path.join(tempfile.gettempdir(), f"ds_config_{rank}.json")
        with open(temp_ds_config, 'w') as f:
            json.dump(ds_config, f)
        
        # 添加DeepSpeed相关参数
        cmd_args.extend([
            '--deepspeed_config', temp_ds_config
        ])
    
    # 打印完整命令供调试
    full_cmd = [sys.executable, 'app.py'] + cmd_args
    print(f"进程 {rank} 执行命令: {' '.join(full_cmd)}")
    
    try:
        # 执行命令
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出进程日志
        last_lines = {}  # 为每个进度类型记录最后一行日志，用于去重
        for line in iter(process.stdout.readline, ''):
            # 过滤掉页面分割和文件解析等冗长日志消息
            should_log = True
            
            # 需要过滤的日志类型
            skip_patterns = [
                "Splitting on page-break",  # 页面分割日志
                "Split into",               # 分割结果日志
                "Parsing OEBPS/Text",       # XHTML解析日志
                "pages to extract: [",      # 页面提取信息
                "Converting page",          # 页面转换信息
                "Chapter extraction progress:"  # 章节提取进度
            ]
            
            # 检查是否需要跳过此日志
            for pattern in skip_patterns:
                if pattern in line:
                    should_log = False
                    break
            
            # 增强的去重逻辑
            line_content = line.strip()
            
            # 特别处理转换进度日志
            if "Converting" in line_content:
                # 提取进度百分比和数字作为键
                progress_key = ""
                match = re.search(r'Converting ([\d.]+)%: : (\d+)', line_content)
                if match:
                    progress_key = f"Converting_{match.group(1)}_{match.group(2)}"
                    
                    # 检查是否是重复的转换进度
                    if progress_key in last_lines:
                        should_log = False
                    else:
                        last_lines[progress_key] = True
                        # 限制字典大小，防止内存泄漏
                        if len(last_lines) > 1000:
                            # 只保留最新的500个条目
                            keys_to_remove = list(last_lines.keys())[:-500]
                            for key in keys_to_remove:
                                last_lines.pop(key, None)
            
            # 只输出应该记录的日志
            if should_log:
                print(f"[进程 {rank}] {line.strip()}")
        
        # 等待进程完成
        return_code = process.wait()
        print(f"进程 {rank} 完成，返回码: {return_code}")
        
    except Exception as e:
        print(f"进程 {rank} 执行出错: {e}")
    finally:
        # 清理分布式环境
        cleanup_distributed()
        
        # 清理临时文件，只清理我们创建的临时配置文件
        if ('--tts_engine' in cmd_args and cmd_args[cmd_args.index('--tts_engine') + 1] == 'xtts') and '--deepspeed_config' not in args:
            if 'temp_ds_config' in locals() and os.path.exists(temp_ds_config):
                os.remove(temp_ds_config)
                print(f"已删除临时配置文件: {temp_ds_config}")

def cleanup_temp_files():
    """清理遗留的临时文件"""
    for pattern in ['*_temp_*', 'ds_config_*.json']:
        for file_path in glob.glob(os.path.join(tempfile.gettempdir(), pattern)):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
                else:
                    os.remove(file_path)
                print(f"已删除临时文件: {file_path}")
            except Exception as e:
                print(f"无法删除临时文件 {file_path}: {e}")

def signal_handler(sig, frame):
    """处理终止信号"""
    print("接收到中断信号，正在清理并退出...")
    cleanup_distributed()
    cleanup_temp_files()
    sys.exit(0)

def main():
    """主函数，处理命令行参数并启动多进程"""
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 清理之前可能存在的临时文件
    cleanup_temp_files()
    
    # 检测可用的GPU
    num_gpus = get_num_gpus()
    if num_gpus == 0:
        print("未检测到可用的GPU，无法进行多GPU处理")
        sys.exit(1)
    
    print(f"检测到 {num_gpus} 个可用的GPU")
    
    # 添加分布式支持到命令行参数
    parser = argparse.ArgumentParser(description='启动多GPU处理')
    parser.add_argument('--nproc_per_node', type=int, default=num_gpus,
                        help='总进程数（默认等于GPU数量）')
    parser.add_argument('--procs_per_gpu', type=int, default=1,
                        help='每个GPU上运行的进程数（默认为1）')
    parser.add_argument('--balance_workload', type=int, default=1,
                        help='是否启用工作负载均衡（0=关闭，1=启用，默认1）')
    
    # 解析我们自己的参数
    my_args, app_args = parser.parse_known_args()
    
    # 计算每个GPU的进程数
    procs_per_gpu = my_args.procs_per_gpu
    
    # 如果指定了总进程数但没有指定每GPU进程数，则计算每GPU进程数
    if my_args.nproc_per_node > num_gpus and procs_per_gpu == 1:
        procs_per_gpu = my_args.nproc_per_node // num_gpus
        print(f"根据总进程数计算得到每GPU进程数: {procs_per_gpu}")
    
    # 计算总进程数
    total_procs = num_gpus * procs_per_gpu
    
    # 使用指定的总进程数（如果提供）
    if my_args.nproc_per_node != num_gpus:
        total_procs = my_args.nproc_per_node
        print(f"使用指定的总进程数: {total_procs}")
    
    # 获取可用的GPU IDs
    gpu_ids = list(range(num_gpus))
    
    # 如果启用了工作负载均衡，则在命令行参数中添加均衡标志
    if my_args.balance_workload == 1:
        for i, arg in enumerate(app_args):
            if arg == '--ebook' and i + 1 < len(app_args):
                # 添加工作负载均衡参数，确保每个进程处理电子书的不同部分
                app_args.extend(['--balance_work', 'True'])
                print("已启用工作负载均衡，将分配电子书的不同部分给每个进程处理")
                break
            elif arg == '--ebooks_dir' and i + 1 < len(app_args):
                # 对于多本书情况，每个进程处理不同的书
                app_args.extend(['--parallel_books', 'True'])
                print("已启用多书并行处理，将不同的书分配给不同的进程")
                break
    
    print(f"将使用 {num_gpus} 个GPU，每个GPU {procs_per_gpu} 个进程，总共 {total_procs} 个进程进行处理")
    
    # 设置多处理方法为spawn（必须为多GPU支持）
    mp.set_start_method('spawn', force=True)
    
    # 创建进程
    processes = []
    for rank in range(total_procs):
        p = mp.Process(
            target=run_worker,
            args=(rank, total_procs, app_args, gpu_ids, procs_per_gpu)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 清理临时文件
    cleanup_temp_files()
    
    print("所有进程已完成")

if __name__ == "__main__":
    main() 