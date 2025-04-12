#!/usr/bin/env python
"""
多GPU启动脚本 - 为ebook2audiobook启用分布式处理
"""

import os
import sys
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
    parser.add_argument("--nproc_per_node", type=int, default=4,
                      help="每个节点上的进程数（默认：4）")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1",
                      help="主节点地址（默认：127.0.0.1）")
    parser.add_argument("--master_port", type=str, default="29500",
                      help="主节点端口（默认：29500）")
    
    # 将剩余参数传递给app.py
    parser.add_argument('app_args', nargs='*', 
                      help='传递给应用程序的参数')
    
    return parser.parse_args()

def run_app(rank, world_size, args):
    """运行ebook2audiobook应用"""
    # 配置进程环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(args.node_rank * args.nproc_per_node + rank)
    os.environ['LOCAL_RANK'] = str(rank)
    
    print(f"启动进程: rank={rank}, world_size={world_size}")
    
    # 导入并运行app
    cmd = [sys.executable, "app.py"]
    cmd.extend(args.app_args)
    cmd.append(f"--gpu_id={rank}")
    
    # 启动进程
    os.system(" ".join(cmd))

def main():
    """主函数"""
    args = parse_args()
    
    # 计算总进程数
    world_size = args.nnodes * args.nproc_per_node
    
    if world_size > 1:
        print(f"正在启动分布式训练: {args.nproc_per_node} 进程 x {args.nnodes} 节点")
        # 使用torch.multiprocessing启动多个进程
        mp.spawn(
            run_app,
            args=(world_size, args),
            nprocs=args.nproc_per_node,
            join=True
        )
    else:
        print("单进程模式启动")
        run_app(0, 1, args)

if __name__ == "__main__":
    main() 