#!/usr/bin/env python
"""
多GPU启动脚本 - 为ebook2audiobook启用分布式处理
"""

import os
import sys
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from argparse import ArgumentParser

def parse_args():
    """解析命令行参数"""
    parser = ArgumentParser(description="多GPU启动器 for ebook2audiobook")
    parser.add_argument("--nnodes", type=int, default=1, 
                      help="节点数量（默认：1）")
    parser.add_argument("--node_rank", type=int, default=0, 
                      help="当前节点排名（默认：0）")
    parser.add_argument("--nproc_per_node", type=int, default=None,
                      help="每个节点上的进程数（默认：自动检测GPU数量）")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1",
                      help="主节点地址（默认：127.0.0.1）")
    parser.add_argument("--master_port", type=str, default="29500",
                      help="主节点端口（默认：29500）")
    
    # 将剩余参数传递给app.py
    parser.add_argument('app_args', nargs='*', 
                      help='传递给应用程序的参数')
    
    return parser.parse_args()

def print_gpu_info():
    """打印GPU信息"""
    try:
        gpu_count = torch.cuda.device_count()
        print(f"系统检测到 {gpu_count} 个GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # 转换为GB
            print(f"GPU {i}: {gpu_name}, 显存: {total_mem:.2f} GB")
    except Exception as e:
        print(f"获取GPU信息失败: {e}")

def run_worker(rank, world_size, args):
    """每个工作进程的入口点"""
    print(f"启动进程: rank={rank}, world_size={world_size}")
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(args.node_rank * args.nproc_per_node + rank)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # 明确设置当前进程使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    # 打印当前进程的环境和GPU信息
    print(f"进程 {rank} 的环境变量:")
    print(f"  - MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"  - MASTER_PORT: {os.environ['MASTER_PORT']}")
    print(f"  - WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    print(f"  - RANK: {os.environ['RANK']}")
    print(f"  - LOCAL_RANK: {os.environ['LOCAL_RANK']}")
    print(f"  - CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # 构建命令
    cmd = [sys.executable, "app.py"]
    cmd.extend(args.app_args)
    
    # 添加特定于分布式训练的参数
    cmd.append(f"--gpu_id={rank}")
    cmd.append("--use_distributed=True")
    
    # 将命令转换为字符串并执行
    cmd_str = " ".join(cmd)
    print(f"进程 {rank} 执行: {cmd_str}")
    
    try:
        # 使用subprocess执行，可以捕获输出
        process = subprocess.Popen(
            cmd_str, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时输出进程日志
        for line in process.stdout:
            sys.stdout.write(f"[进程 {rank}] {line}")
            sys.stdout.flush()
            
        # 等待进程完成
        returncode = process.wait()
        print(f"进程 {rank} 退出，返回代码: {returncode}")
        
    except Exception as e:
        print(f"进程 {rank} 执行出错: {e}")
    
    return 0

def main():
    """主函数"""
    args = parse_args()
    
    # 打印GPU信息
    print_gpu_info()
    
    # 确定GPU和进程数量
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        print("错误: 未检测到GPU，无法执行多GPU处理")
        sys.exit(1)
    
    if args.nproc_per_node is None:
        args.nproc_per_node = gpu_count
    
    world_size = args.nnodes * args.nproc_per_node
    
    print(f"启动分布式执行: {args.nproc_per_node} 进程 x {args.nnodes} 节点 = {world_size} 总进程")
    
    # 使用多进程启动多个worker
    if world_size > 1:
        print(f"使用torch.multiprocessing.spawn启动 {args.nproc_per_node} 个进程")
        mp.spawn(
            run_worker,
            args=(world_size, args),
            nprocs=args.nproc_per_node,
            join=True
        )
    else:
        print("只有一个GPU，使用单进程模式")
        run_worker(0, 1, args)

if __name__ == "__main__":
    main() 