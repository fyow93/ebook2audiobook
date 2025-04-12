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
    
    # 获取系统中所有可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"进程 {rank} 检测到系统中有 {num_gpus} 个GPU")
    
    # 确保所有GPU对于所有进程都是可见的，而不是每个进程只能看到一个GPU
    # 我们不设置CUDA_VISIBLE_DEVICES，而是让所有进程看到所有GPU
    # 然后在代码中明确指定使用哪个GPU
    
    # 设置环境变量 
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(args.node_rank * args.nproc_per_node + rank)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # 打印当前进程的环境和GPU信息
    print(f"进程 {rank} 的环境变量:")
    print(f"  - MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"  - MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"  - WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"  - RANK: {os.environ.get('RANK')}")
    print(f"  - LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    
    # 构建命令
    cmd = [sys.executable, "app.py"]
    cmd.extend(args.app_args)
    
    # 添加特定于分布式训练的参数
    cmd.append(f"--gpu_id={rank}")
    cmd.append("--use_distributed=True")
    cmd.append("--device=cuda")  # 强制使用CUDA
    
    # 将命令转换为字符串并执行
    cmd_str = " ".join(cmd)
    print(f"进程 {rank} 执行: {cmd_str}")
    
    try:
        # 初始化进程组（如果需要）
        if rank == 0:
            print(f"进程 {rank} 初始化分布式环境...")
            if not dist.is_initialized():
                try:
                    dist.init_process_group(
                        backend="nccl",
                        init_method=f"tcp://{args.master_addr}:{args.master_port}",
                        world_size=world_size,
                        rank=int(os.environ['RANK'])
                    )
                    print(f"进程 {rank} 成功初始化分布式环境")
                except Exception as e:
                    print(f"进程 {rank} 初始化分布式环境失败: {e}")
        
        # 使用subprocess执行，可以捕获输出
        process = subprocess.Popen(
            cmd_str, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=os.environ.copy()  # 确保环境变量传递给子进程
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
    
    # 清理分布式环境
    if dist.is_initialized():
        dist.destroy_process_group()
    
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
        print(f"自动设置进程数为GPU数量: {args.nproc_per_node}")
    
    world_size = args.nnodes * args.nproc_per_node
    
    # 清理旧的分布式文件
    import glob
    for f in glob.glob("/tmp/torch_distributed_*"):
        try:
            os.remove(f)
            print(f"删除临时文件: {f}")
        except:
            pass
    
    print(f"启动分布式执行: {args.nproc_per_node} 进程 x {args.nnodes} 节点 = {world_size} 总进程")
    
    # 设置正确的分布式环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # 使用多进程启动多个worker
    if world_size > 1:
        print(f"使用torch.multiprocessing.spawn启动 {args.nproc_per_node} 个进程")
        mp.set_start_method('spawn', force=True)
        try:
            mp.spawn(
                run_worker,
                args=(world_size, args),
                nprocs=args.nproc_per_node,
                join=True
            )
        except KeyboardInterrupt:
            print("用户中断，正在结束所有进程...")
    else:
        print("只有一个GPU，使用单进程模式")
        run_worker(0, 1, args)

if __name__ == "__main__":
    main() 