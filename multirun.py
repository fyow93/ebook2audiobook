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
import json
import datetime

# 进度收集相关的常量
PROGRESS_DIR = os.path.join(tempfile.gettempdir(), "ebook2audiobook_progress")
PROGRESS_UPDATE_INTERVAL = 10  # 10秒更新一次总进度

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
    os.environ['LOCAL_RANK'] = str(rank % get_num_gpus())  # 确保LOCAL_RANK不超过实际GPU数量
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # 初始化进程组
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)  # 使用gloo后端替代nccl，更稳定
        print(f"分布式进程 {rank}/{world_size} 初始化成功")
        
        # 设置当前设备，确保不超出GPU数量
        gpu_id = rank % get_num_gpus()
        torch.cuda.set_device(gpu_id)
        
        return True
    except Exception as e:
        print(f"初始化分布式环境失败: {e}")
        return False

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("已销毁分布式进程组")

def cleanup_progress_files():
    """清理进度文件"""
    if os.path.exists(PROGRESS_DIR):
        try:
            shutil.rmtree(PROGRESS_DIR)
            print(f"已清理进度文件目录: {PROGRESS_DIR}")
        except Exception as e:
            print(f"清理进度文件目录失败: {e}")

def run_worker(rank, world_size, args, gpu_ids, procs_per_gpu):
    """每个工作进程的执行函数"""
    # 设置分布式环境
    success = setup_distributed(rank, world_size)
    if not success:
        print(f"进程 {rank} 无法初始化分布式环境，退出")
        return
    
    # 计算当前进程应该使用的GPU ID - 确保不超出实际GPU数量
    num_gpus = len(gpu_ids)
    gpu_id = gpu_ids[rank % num_gpus]
    proc_id_on_gpu = rank // num_gpus if (rank // num_gpus) < procs_per_gpu else rank % procs_per_gpu
    
    print(f"工作进程 {rank} 使用 GPU ID: {gpu_id} (GPU {gpu_id} 上的第{proc_id_on_gpu + 1}个进程，共{procs_per_gpu}个)")
    
    # 设置DeepSpeed相关环境变量
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__)) + ":" + os.environ.get('PYTHONPATH', '')

    # 准备命令行参数
    cmd_args = args.copy()
    
    # 添加或修改为分布式模式必须的参数
    cmd_args.extend([
        '--gpu_id', str(gpu_id),
        '--use_distributed', 'True'
    ])
    
    # 如果命令行中未指定deepspeed_config，且需要deepspeed支持
    temp_ds_config = None
    if '--deepspeed_config' not in cmd_args:
        cmd_args.append('--deepspeed')
    
    # 实现工作分片 - 添加处理范围控制
    # 使用rank和world_size来确定每个进程应该处理的范围
    if '--ebook' in cmd_args:
        # 针对电子书处理，添加范围参数
        cmd_args.extend([
            '--process_rank', str(rank),
            '--total_processes', str(world_size)
        ])
        print(f"进程 {rank}/{world_size} 将只处理其对应的工作分片")
    
    # 检查是否已经有--headless参数
    if '--headless' not in cmd_args:
        cmd_args.append('--headless')
        
    # 打印完整命令供调试
    full_cmd = [sys.executable, 'app.py'] + cmd_args
    print(f"进程 {rank} 执行命令: {' '.join(full_cmd)}")
    
    # 为进程0创建进度收集目录
    if rank == 0:
        os.makedirs(PROGRESS_DIR, exist_ok=True)
        print(f"进程0已创建进度收集目录: {PROGRESS_DIR}")
        # 清理可能存在的旧进度文件
        for old_file in glob.glob(os.path.join(PROGRESS_DIR, "progress_*.json")):
            try:
                os.remove(old_file)
            except:
                pass
    
    try:
        # 执行命令
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 进度信息
        progress_info = {
            "rank": rank,
            "current_chapter": 0,
            "chapter_range": "",
            "processed_sentences": 0,
            "total_sentences": 0,
            "last_update": time.time()
        }
        
        last_progress_write = 0
        last_progress_display = 0
        
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
                match = re.search(r'Converting ([\d.]+)%: : (\d+)/(\d+)', line_content)
                if match:
                    progress_key = f"Converting_{match.group(1)}_{match.group(2)}"
                    
                    # 捕获转换总数和当前进度
                    processed = int(match.group(2))
                    total = int(match.group(3))
                    
                    # 更新进度信息
                    progress_info["processed_sentences"] = processed
                    progress_info["total_sentences"] = total
                    
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
            
            # 捕获章节范围和当前章节信息
            chapter_range_match = re.search(r'进程 \d+/\d+ 将处理章节 (\d+) 到 (\d+) \(共(\d+)章\)', line_content)
            if chapter_range_match:
                start_chapter = chapter_range_match.group(1)
                end_chapter = chapter_range_match.group(2)
                total_chapters = chapter_range_match.group(3)
                progress_info["chapter_range"] = f"{start_chapter}-{end_chapter}/{total_chapters}"
            
            current_chapter_match = re.search(r'处理章节 (\d+)', line_content)
            if current_chapter_match:
                progress_info["current_chapter"] = int(current_chapter_match.group(1))
            
            # 更新和写入进度信息
            current_time = time.time()
            if current_time - last_progress_write >= 2:  # 每2秒写入一次进度
                progress_info["last_update"] = current_time
                progress_file = os.path.join(PROGRESS_DIR, f"progress_{rank}.json")
                
                # 安全写入进度文件
                try:
                    temp_file = f"{progress_file}.tmp"
                    with open(temp_file, 'w') as f:
                        json.dump(progress_info, f)
                    # 原子重命名，确保文件写入完整
                    os.replace(temp_file, progress_file)
                except Exception as e:
                    print(f"写入进度文件失败: {e}")
                
                last_progress_write = current_time
            
            # 进程0负责汇总并显示所有进程的进度
            if rank == 0 and current_time - last_progress_display >= PROGRESS_UPDATE_INTERVAL:
                collect_and_display_progress(world_size)
                last_progress_display = current_time
            
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
        
        # 清理临时文件
        if temp_ds_config and os.path.exists(temp_ds_config):
            try:
                os.remove(temp_ds_config)
                print(f"已删除临时配置文件: {temp_ds_config}")
            except Exception as e:
                print(f"无法删除临时文件 {temp_ds_config}: {e}")
        
        # 删除自己的进度文件
        progress_file = os.path.join(PROGRESS_DIR, f"progress_{rank}.json")
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
            except:
                pass

def collect_and_display_progress(world_size):
    """收集并显示所有进程的进度"""
    if not os.path.exists(PROGRESS_DIR):
        return
    
    try:
        progress_files = glob.glob(os.path.join(PROGRESS_DIR, "progress_*.json"))
        if not progress_files:
            return
        
        all_progress = []
        total_processed = 0
        total_sentences = 0
        
        # 收集所有进程的进度
        for progress_file in progress_files:
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    all_progress.append(progress)
                    total_processed += progress["processed_sentences"]
                    
                    # 只从一个进程获取总句子数，因为所有进程看到的总数应该相同
                    if progress["total_sentences"] > total_sentences:
                        total_sentences = progress["total_sentences"]
            except Exception as e:
                print(f"读取进度文件 {progress_file} 失败: {e}")
        
        if not all_progress or total_sentences == 0:
            return
        
        # 按进程编号排序
        all_progress.sort(key=lambda x: x["rank"])
        
        # 计算总体进度
        percentage = (total_processed / total_sentences) * 100 if total_sentences > 0 else 0
        
        # 计算处理速度和预计剩余时间
        now = time.time()
        times = [p["last_update"] for p in all_progress]
        if times:
            oldest_update = min(times)
            elapsed_seconds = now - oldest_update
            if elapsed_seconds > 0:
                sentences_per_second = total_processed / elapsed_seconds
                remaining_sentences = total_sentences - total_processed
                eta_seconds = remaining_sentences / sentences_per_second if sentences_per_second > 0 else 0
                
                # 格式化ETA为小时:分钟:秒
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_seconds = int(eta_seconds % 60)
                eta_str = f"{eta_hours}h {eta_minutes}m {eta_seconds}s"
                
                # 打印总进度
                print(f"\n总进度: {percentage:.2f}% [{total_processed}/{total_sentences}] 速度: {sentences_per_second:.2f}句/秒 ETA: {eta_str}")
                
                # 打印表头
                print(f"{'进程':<6}| {'章节范围':<18}| {'当前章节':<10}| {'进度'}")
                print(f"{'-'*6}-+-{'-'*18}-+-{'-'*10}-+-{'-'*15}")
                
                # 打印每个进程的进度
                for p in all_progress:
                    rank = p["rank"]
                    chapter_range = p["chapter_range"]
                    current_chapter = p["current_chapter"]
                    proc_sentences = p["processed_sentences"]
                    print(f" {rank:<5}| {chapter_range:<18}| 章节 {current_chapter:<5}| {proc_sentences}/{total_sentences} 句")
                
                print("")  # 空行分隔
    except Exception as e:
        print(f"收集和显示进度时发生错误: {e}")

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
    
    # 同时清理进度文件
    cleanup_progress_files()

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
    parser.add_argument('--output_dir', type=str, default="output",
                        help='输出文件的汇总目录（默认为当前目录下的output）')
    parser.add_argument('--progress_interval', type=int, default=10,
                        help='进度更新间隔（秒，默认10秒）')
    
    # 解析我们自己的参数
    my_args, app_args = parser.parse_known_args()
    
    # 设置进度更新间隔
    global PROGRESS_UPDATE_INTERVAL
    PROGRESS_UPDATE_INTERVAL = my_args.progress_interval
    
    # 计算每个GPU的进程数
    procs_per_gpu = min(my_args.procs_per_gpu, 8)  # 限制每GPU最多8个进程，防止资源耗尽
    
    # 如果指定了总进程数但没有指定每GPU进程数，则计算每GPU进程数
    if my_args.nproc_per_node > num_gpus and procs_per_gpu == 1:
        procs_per_gpu = min(my_args.nproc_per_node // num_gpus, 8)  # 限制最大进程数
        print(f"根据总进程数计算得到每GPU进程数: {procs_per_gpu}")
    
    # 计算总进程数，确保不超过合理的上限
    total_procs = min(num_gpus * procs_per_gpu, 32)  # 限制总进程数不超过32
    
    # 使用指定的总进程数（如果提供），但确保不会过大
    if my_args.nproc_per_node != num_gpus:
        total_procs = min(my_args.nproc_per_node, 32)  # 限制总进程数
        print(f"使用指定的总进程数: {total_procs}")
    
    # 获取可用的GPU IDs
    gpu_ids = list(range(num_gpus))
    
    # 注意：以下代码已被注释，因为app.py尚未实现balance_work参数
    # 工作均衡将通过进程ID到GPU ID的映射隐式处理
    """
    # 如果启用了工作负载均衡，则在命令行参数中添加均衡标志
    if my_args.balance_workload == 1:
        balance_added = False
        for i, arg in enumerate(app_args):
            if arg == '--ebook' and i + 1 < len(app_args):
                # 添加工作负载均衡参数，确保每个进程处理电子书的不同部分
                if '--balance_work' not in app_args:
                    app_args.extend(['--balance_work', 'True'])
                    print("已启用工作负载均衡，将分配电子书的不同部分给每个进程处理")
                    balance_added = True
                break
            elif arg == '--ebooks_dir' and i + 1 < len(app_args):
                # 对于多本书情况，每个进程处理不同的书
                if '--parallel_books' not in app_args:
                    app_args.extend(['--parallel_books', 'True'])
                    print("已启用多书并行处理，将不同的书分配给不同的进程")
                    balance_added = True
                break
        
        # 如果没有找到ebook或ebooks_dir参数，但依然希望启用均衡
        if not balance_added:
            app_args.extend(['--balance_work', 'True'])
            print("已启用通用工作负载均衡模式")
    """
    
    if my_args.balance_workload == 1:
        print("工作负载均衡：使用进程ID到GPU ID的映射来隐式平衡负载（无需app.py支持）")
    
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
    
    # 创建输出目录（如果不存在）
    output_dir = my_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在将所有音频文件汇总到 {output_dir} 目录...")
    
    # 寻找并复制所有音频文件
    collect_audio_files(output_dir)
    
    print(f"音频文件汇总完成，请查看 {output_dir} 目录")

def collect_audio_files(output_dir):
    """收集所有音频文件到指定的输出目录"""
    try:
        # 查找tmp目录
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        if not os.path.exists(tmp_dir):
            print(f"警告: 未找到临时目录 {tmp_dir}")
            return
            
        # 搜索所有可能的音频文件格式
        audio_formats = ['.flac', '.mp3', '.wav', '.m4a', '.m4b', '.ogg', '.aac']
        copied_files = 0
        
        # 创建日志文件来跟踪复制的文件
        log_file = os.path.join(output_dir, "audio_files_log.txt")
        
        with open(log_file, "w", encoding="utf-8") as log:
            # 递归搜索tmp目录下的所有音频文件
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_formats):
                        # 找到音频文件
                        src_path = os.path.join(root, file)
                        
                        # 创建目标路径，保持相对于tmp目录的相对路径结构
                        rel_path = os.path.relpath(root, tmp_dir)
                        if rel_path != ".":
                            # 如果是子目录，创建相应的目录结构
                            dest_dir = os.path.join(output_dir, rel_path)
                            os.makedirs(dest_dir, exist_ok=True)
                            dest_path = os.path.join(dest_dir, file)
                        else:
                            # 如果是根目录，直接放在输出目录中
                            dest_path = os.path.join(output_dir, file)
                        
                        # 复制文件
                        try:
                            shutil.copy2(src_path, dest_path)
                            copied_files += 1
                            log.write(f"已复制: {src_path} -> {dest_path}\n")
                        except Exception as e:
                            log.write(f"复制失败: {src_path} -> {dest_path}, 错误: {str(e)}\n")
        
        print(f"共复制了 {copied_files} 个音频文件到 {output_dir} 目录")
        print(f"详细日志请查看: {log_file}")
        
    except Exception as e:
        print(f"收集音频文件时出错: {str(e)}")

if __name__ == "__main__":
    main() 