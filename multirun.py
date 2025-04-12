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

def run_worker(rank, world_size, args, gpu_ids):
    """每个工作进程的执行函数"""
    # 设置分布式环境
    success = setup_distributed(rank, world_size)
    if not success:
        print(f"进程 {rank} 无法初始化分布式环境，退出")
        return
    
    # 获取当前进程的GPU ID
    gpu_id = gpu_ids[rank] if rank < len(gpu_ids) else rank
    
    print(f"工作进程 {rank} 使用 GPU ID: {gpu_id}")
    
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
    
    # 检查是否已经有--headless参数
    if '--headless' not in cmd_args:
        cmd_args.append('--headless')
        
    # 如果有需要，可以添加DeepSpeed特定的参数
    if '--tts_engine' not in cmd_args or ('--tts_engine' in cmd_args and cmd_args[cmd_args.index('--tts_engine') + 1] == 'xtts'):
        # 为DeepSpeed创建临时配置文件
        ds_config = {
            "train_batch_size": 1,
            "fp16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu"}
            },
            "gradient_accumulation_steps": 1
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
        for line in iter(process.stdout.readline, ''):
            print(f"[进程 {rank}] {line.strip()}")
        
        # 等待进程完成
        return_code = process.wait()
        print(f"进程 {rank} 完成，返回码: {return_code}")
        
    except Exception as e:
        print(f"进程 {rank} 执行出错: {e}")
    finally:
        # 清理分布式环境
        cleanup_distributed()
        
        # 清理临时文件
        if '--tts_engine' in cmd_args and cmd_args[cmd_args.index('--tts_engine') + 1] == 'xtts':
            if os.path.exists(temp_ds_config):
                os.remove(temp_ds_config)

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
                        help='每个节点使用的进程数（通常等于GPU数量）')
    
    # 解析我们自己的参数
    my_args, app_args = parser.parse_known_args()
    
    # 确保进程数不超过GPU数量
    nproc_per_node = min(my_args.nproc_per_node, num_gpus)
    
    # 获取可用的GPU IDs
    gpu_ids = list(range(num_gpus))
    
    print(f"将使用 {nproc_per_node} 个进程进行处理")
    
    # 设置多处理方法为spawn（必须为多GPU支持）
    mp.set_start_method('spawn', force=True)
    
    # 创建进程
    processes = []
    for rank in range(nproc_per_node):
        p = mp.Process(
            target=run_worker,
            args=(rank, nproc_per_node, app_args, gpu_ids)
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