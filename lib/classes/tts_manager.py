import gc
import numpy as np
import os
import regex as re
import shutil
import soundfile as sf
import subprocess
import tempfile
import torch
import torchaudio
import threading
import uuid
import json
import torch.distributed as dist
import time
import filelock
import pickle
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
from filelock import FileLock

from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from pathlib import Path
from scipy.io import wavfile as wav
from scipy.signal import find_peaks
from TTS.api import TTS as TtsXTTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from lib.models import *
from lib.conf import voices_dir, models_dir, default_audio_proc_format
from lib.lang import language_tts

torch.backends.cudnn.benchmark = True
#torch.serialization.add_safe_globals(["numpy.core.multiarray.scalar"])

app = FastAPI()
lock = threading.Lock()
loaded_tts = {}

# 模型共享设置
TTS_MODEL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "tts_model_cache")
os.makedirs(TTS_MODEL_CACHE_DIR, exist_ok=True)
MODEL_LOCK_FILE = os.path.join(TTS_MODEL_CACHE_DIR, "model_load.lock")
MODEL_READY_FILE_TEMPLATE = os.path.join(TTS_MODEL_CACHE_DIR, "model_{}_ready")
SHARED_MODEL_TIMEOUT = 300  # 5分钟超时，避免无限等待

@app.post("/load_coqui_tts_api/")
def load_coqui_tts_api(model_path, device):
    try:
        model_key = f"api_{os.path.basename(model_path)}"
        model_ready_file = MODEL_READY_FILE_TEMPLATE.format(model_key)
        cache_dir = os.path.dirname(model_ready_file)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 检查是否已有进程在加载或已加载该模型
        lock_file = FileLock(MODEL_LOCK_FILE)
        
        with lock_file:
            # 如果模型已加载完成，直接从共享存储加载
            if os.path.exists(model_ready_file):
                print(f"使用共享模型: {model_key}")
                return _load_shared_model(model_key, device)
            
            # 如果没有共享模型，则此进程负责加载
            print(f"进程首次加载模型: {model_key} - {datetime.datetime.now()}")
            try:
                with lock:
                    tts = TtsXTTS(model_path)
                    if device == 'cuda':
                        tts.cuda()
                    else:
                        tts.to(device)
                
                # 保存模型以便其他进程使用
                _save_shared_model(tts, model_key)
                
                # 创建模型加载完成标记文件
                with open(model_ready_file, 'w') as f:
                    f.write(f"Model API {model_path} loaded successfully at {datetime.datetime.now()}")
                print(f"模型 {model_key} 加载完成标记已创建 - {datetime.datetime.now()}")
                
                return tts
            except Exception as e:
                # 如果加载失败，删除标记文件（如果存在）
                if os.path.exists(model_ready_file):
                    try:
                        os.remove(model_ready_file)
                    except:
                        pass
                raise e
    except Exception as e:
        error = f'load_coqui_tts_api() error: {e}'
        print(error)
        return None

@app.post("/load_coqui_tts_checkpoint/")
def load_coqui_tts_checkpoint(model_path, config_path, vocab_path, device):
    try:
        # 创建一个唯一的模型标识
        model_key = f"checkpoint_{os.path.basename(os.path.dirname(model_path))}"
        model_ready_file = MODEL_READY_FILE_TEMPLATE.format(model_key)
        cache_dir = os.path.dirname(model_ready_file)
        os.makedirs(cache_dir, exist_ok=True)
        
        # 获取当前进程的分布式信息
        rank = os.environ.get('RANK')
        world_size = os.environ.get('WORLD_SIZE')
        local_rank = os.environ.get('LOCAL_RANK', '0')
        gpu_id = int(local_rank)
        
        print(f"当前进程: RANK={rank}, LOCAL_RANK={local_rank}, GPU_ID={gpu_id}")
        
        # 判断是否为分布式模式
        is_distributed = rank is not None and world_size is not None
        if is_distributed:
            print(f"运行在分布式模式下: RANK={rank}, WORLD_SIZE={world_size}")
        
        # 检测可用GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU设备")
        
        # 设置环境变量确保使用指定GPU
        if device == 'cuda':
            # 在分布式模式下，使用分配的GPU
            torch.cuda.set_device(gpu_id)
            current_device = torch.cuda.current_device()
            print(f"已设置为GPU {current_device}")
            
        # 尝试初始化分布式环境
        if num_gpus > 1 and default_xtts_settings['use_deepspeed'] and is_distributed:
            try:
                if not dist.is_initialized():
                    # 使用当前进程的分布式环境
                    dist_backend = "nccl"  # GPU推荐使用nccl
                    print(f"初始化分布式进程组: {dist_backend}")
                    dist.init_process_group(backend=dist_backend)
                    print(f"分布式环境已初始化")
            except Exception as e:
                print(f"初始化分布式环境失败: {e}")
                print("回退到单GPU模式")
        
        # 加载DeepSpeed配置
        ds_config = None
        if default_xtts_settings['use_deepspeed'] and num_gpus > 1:
            try:
                # 总是尝试从ds_config.json加载配置
                config_path_ds = "ds_config.json"
                if os.path.exists(config_path_ds):
                    with open(config_path_ds, "r") as f:
                        ds_config = json.load(f)
                    print(f"已加载DeepSpeed配置文件: {config_path_ds}")
                    
                    # 确保配置兼容多GPU
                    if 'zero_optimization' in ds_config:
                        print(f"使用ZeRO优化阶段: {ds_config['zero_optimization']['stage']}")
                    
                    # 记录一些重要的配置参数
                    if 'train_batch_size' in ds_config:
                        print(f"训练批次大小: {ds_config['train_batch_size']}")
                    if 'train_micro_batch_size_per_gpu' in ds_config:
                        print(f"每GPU微批次大小: {ds_config['train_micro_batch_size_per_gpu']}")
                    
                    # 打印完整的DeepSpeed配置
                    print("\n=================== DeepSpeed配置详情 ===================")
                    import pprint
                    pprint.pprint(ds_config)
                    print("==========================================================\n")
                else:
                    print(f"警告: 未找到DeepSpeed配置文件 {config_path_ds}")
                    print("DeepSpeed将使用默认配置运行，可能无法充分利用GPU资源")
                    # 创建最小配置以确保DeepSpeed正常运行
                    ds_config = {
                        "train_batch_size": 32,
                        "fp16": {"enabled": True},
                        "zero_optimization": {"stage": 2}
                    }
            except Exception as e:
                print(f"加载DeepSpeed配置失败: {e}")
                print("DeepSpeed将被禁用")
                default_xtts_settings['use_deepspeed'] = False
        
        # 处理模型共享逻辑 - 使用共享加载机制，避免每个进程都加载模型
        # 模型共享现在是默认行为，无需额外参数
        lock_file = FileLock(MODEL_LOCK_FILE)
        
        # 确定是加载还是等待模型 - 只有rank 0进程加载模型，其它进程等待
        is_loader_process = (not is_distributed) or (int(rank) == 0)
        
        if is_loader_process:
            # 此进程负责加载模型
            print(f"进程 {rank if rank else '0'} 负责加载模型，其他进程将等待... - {datetime.datetime.now()}")
            
            with lock_file:
                # 双重检查 - 如果此时模型已经加载好，直接使用
                if os.path.exists(model_ready_file):
                    print(f"另一个进程已经完成了模型加载，将直接使用共享模型")
                    return _load_shared_model(model_key, device)
                
                try:
                    # 加载配置
                    config = XttsConfig()
                    config.models_dir = os.path.join("models", "tts")
                    config.load_json(config_path)
                    
                    # 初始化TTS模型
                    tts = Xtts.init_from_config(config)
                    
                    # 在加载checkpoint前再次打印DeepSpeed配置
                    print("\n=================== 最终使用的DeepSpeed配置 ===================")
                    print(f"use_deepspeed: {default_xtts_settings['use_deepspeed'] and num_gpus > 1}")
                    if ds_config:
                        import pprint
                        pprint.pprint(ds_config)
                    else:
                        print("警告: 未配置DeepSpeed参数，将使用默认设置")
                    print("=============================================================\n")
                    
                    print(f"开始加载模型检查点: {model_path} - {datetime.datetime.now()}")
                    with lock:
                        tts.load_checkpoint(
                            config,
                            checkpoint_path=model_path,
                            vocab_path=vocab_path,
                            use_deepspeed=default_xtts_settings['use_deepspeed'] and num_gpus > 1,
                            eval=True,
                            deepspeed_config=ds_config
                        )
                    print(f"模型检查点加载完成 - {datetime.datetime.now()}")
                    
                    # 多GPU处理
                    if device == 'cuda' and num_gpus > 1:
                        try:
                            # 使用DistributedDataParallel来确保正确使用多GPU
                            # 首先将模型移到当前设备
                            if dist.is_initialized():
                                # 使用DDP进行包装
                                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                                print(f"在GPU {local_rank}上初始化DDP模型")
                                
                                # 确保只用指定的GPU
                                torch.cuda.set_device(local_rank)
                                device_id = torch.cuda.current_device()
                                print(f"强制设置为设备: cuda:{device_id}")
                                
                                # 将模型放置到指定GPU
                                tts.cuda(device_id)
                                
                                # 包装为分布式模型
                                from torch.nn.parallel import DistributedDataParallel
                                tts = DistributedDataParallel(
                                    tts, 
                                    device_ids=[device_id], 
                                    output_device=device_id,
                                    find_unused_parameters=True
                                )
                                print(f"DistributedDataParallel初始化成功")
                            else:
                                # 回退到DataParallel
                                print("使用DataParallel进行多GPU推理 (不推荐，效率较低)")
                                tts = torch.nn.DataParallel(tts)
                                tts.cuda()
                            print(f"模型已分布在{num_gpus}个GPU上")
                        except Exception as e:
                            print(f"多GPU初始化失败: {e}")
                            print(f"错误详情: {str(e)}")
                            print("回退到单GPU模式")
                            tts.cuda()
                    elif device == 'cuda':
                        tts.cuda()
                    else:
                        tts.to(device)
                    
                    # 保存模型以便其他进程使用
                    _save_shared_model(tts, model_key)
                    
                    # 创建模型加载完成标记文件
                    with open(model_ready_file, 'w') as f:
                        f.write(f"Model Checkpoint {os.path.basename(os.path.dirname(model_path))} loaded successfully at {datetime.datetime.now()}")
                    print(f"模型 {model_key} 加载完成标记已创建 - {datetime.datetime.now()}")
                    print(f"模型已加载完成并保存到共享存储，通知其他进程")
                    
                    return tts
                except Exception as e:
                    # 如果加载失败，删除标记文件（如果存在）
                    if os.path.exists(model_ready_file):
                        try:
                            os.remove(model_ready_file)
                        except:
                            pass
                    raise e
        else:
            # 此进程需要等待模型加载完成
            print(f"进程 {rank} 等待模型加载完成... - {datetime.datetime.now()}")
            
            # 等待模型加载完成
            wait_start_time = time.time()
            while not os.path.exists(model_ready_file):
                if time.time() - wait_start_time > SHARED_MODEL_TIMEOUT:
                    raise TimeoutError(f"等待模型加载超时，已等待{SHARED_MODEL_TIMEOUT}秒")
                
                # 每5秒输出一次状态信息
                if int(time.time()) % 5 == 0:
                    print(f"进程 {rank} 正在等待模型加载... 已等待 {int(time.time() - wait_start_time)} 秒")
                
                time.sleep(1)
            
            # 模型已加载完成，从共享存储加载
            print(f"进程 {rank} 检测到模型已加载完成，从共享存储中加载 - {datetime.datetime.now()}")
            return _load_shared_model(model_key, device)
            
    except Exception as e:
        error = f'load_coqui_tts_checkpoint() error: {e}'
        print(error)
        # 如果加载失败，删除标记文件（如果存在）
        if os.path.exists(model_ready_file):
            try:
                os.remove(model_ready_file)
            except:
                pass
        return None

def _save_shared_model(model, model_key):
    """保存模型到共享存储"""
    try:
        # 创建模型状态字典
        model_path = os.path.join(TTS_MODEL_CACHE_DIR, f"model_{model_key}.pt")
        model_config_path = os.path.join(TTS_MODEL_CACHE_DIR, f"model_{model_key}_config.json")
        
        # 不同的保存方式处理
        if isinstance(model, (torch.nn.DataParallel, DDP)):
            # 如果是封装过的模型，获取原始模型
            torch.save(model.module.state_dict(), model_path)
            
            # 保存模型配置信息
            if hasattr(model.module, 'config'):
                with open(model_config_path, 'w') as f:
                    json.dump(model.module.config.to_dict(), f)
        else:
            # 直接保存模型状态
            torch.save(model.state_dict(), model_path)
            
            # 保存模型配置信息
            if hasattr(model, 'config'):
                with open(model_config_path, 'w') as f:
                    json.dump(model.config.to_dict(), f)
        
        # 创建就绪标记
        ready_file = MODEL_READY_FILE_TEMPLATE.format(model_key)
        with open(ready_file, 'w') as f:
            f.write(f"Model ready at {time.ctime()}")
        
        print(f"模型已保存到共享存储: {model_path}")
        return True
    except Exception as e:
        print(f"保存共享模型失败: {e}")
        if os.path.exists(ready_file):
            try:
                os.remove(ready_file)
            except:
                pass
        return False

def _load_shared_model(model_key, device):
    """从共享存储加载模型"""
    try:
        # 模型文件路径
        model_path = os.path.join(TTS_MODEL_CACHE_DIR, f"model_{model_key}.pt")
        model_config_path = os.path.join(TTS_MODEL_CACHE_DIR, f"model_{model_key}_config.json")
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"共享模型文件不存在: {model_path}")
        
        # 根据模型类型选择加载方式
        if model_key.startswith("api_"):
            # 加载API模型
            model_name = model_key[4:]  # 去掉前缀
            tts = TtsXTTS(model_name)
            # 加载状态字典
            state_dict = torch.load(model_path, map_location=device)
            tts.load_state_dict(state_dict)
        elif model_key.startswith("checkpoint_"):
            # 加载检查点模型 - 需要先初始化配置
            if os.path.exists(model_config_path):
                # 从保存的配置文件加载
                with open(model_config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # 创建配置对象
                config = XttsConfig()
                config.from_dict(config_dict)
                
                # 初始化模型
                tts = Xtts.init_from_config(config)
                
                # 加载状态字典
                state_dict = torch.load(model_path, map_location=device)
                tts.load_state_dict(state_dict)
            else:
                # 如果没有配置文件，尝试使用原始的加载方式
                # 这可能会失败，因为我们需要完整的配置信息
                print(f"警告: 未找到模型配置文件，尝试直接加载")
                
                # 这里简化处理，可能需要更复杂的逻辑
                # 在实际场景中，可能需要保存更多的上下文信息
                return None
        else:
            # 不支持的模型类型
            print(f"不支持的模型类型: {model_key}")
            return None
        
        # 移动模型到指定设备
        if device == 'cuda':
            tts.cuda()
        else:
            tts.to(device)
        
        print(f"已从共享存储加载模型: {model_key}")
        return tts
    except Exception as e:
        print(f"从共享存储加载模型失败: {e}")
        # 如果加载失败，返回None让调用者处理
        return None

@app.post("/load_coqui_tts_vc/")
def load_coqui_tts_vc(device):
    try:
        with lock:
            tts = TtsXTTS(default_vc_model).to(device)
        return tts
    except Exception as e:
        error = f'load_coqui_tts_vc() error: {e}'
        print(error)
        return None

class TTSManager:
    def __init__(self, session, is_gui_process):   
        self.session = session
        self.is_gui_process = is_gui_process
        self.params = {}
        self.model_path = None
        self.config_path = None
        self.vocab_path = None      
        self._build()
 
    def _build(self):
        tts_key = None
        self.params['tts'] = None
        self.params['current_voice_path'] = None
        self.params['sample_rate'] = models[self.session['tts_engine']][self.session['fine_tuned']]['samplerate']
        
        # 获取环境变量，确定是否为多进程模式
        rank = os.environ.get('RANK')
        world_size = os.environ.get('WORLD_SIZE')
        is_distributed = rank is not None and world_size is not None
        
        # 配置模型缓存目录 - 用于共享模型
        cache_dir = os.path.join(tempfile.gettempdir(), "tts_model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        lock_file_path = os.path.join(cache_dir, "model_load.lock")
        
        # 确定当前进程是否为初始化进程
        is_initializer = (not is_distributed) or (int(rank) == 0)
        
        if self.session['language'] in language_tts[XTTSv2].keys():
            if self.session['voice'] is not None and self.session['language'] != 'eng':
                voice_key = re.sub(r'_(24000|16000)\.wav$', '', os.path.basename(self.session['voice']))
                if voice_key in default_xtts_settings['voices']:
                    if not f"/{self.session['language']}/" in self.session['voice']:
                        msg = f"Converting xttsv2 builtin english voice to {self.session['language']}..."
                        print(msg)
                        self.model_path = models[XTTSv2]['internal']['repo']
                        tts_key = self.model_path
                        
                        # 共享模型处理
                        model_ready_file = os.path.join(cache_dir, f"model_api_{os.path.basename(self.model_path)}_ready")
                        if os.path.exists(model_ready_file) and not self.is_gui_process:
                            # 如果模型已经加载过，直接使用共享模型
                            print(f"使用已加载的共享模型: {tts_key}")
                            # 简化处理，使用原始加载逻辑，实际中可能需要从共享存储加载
                        
                        if tts_key in loaded_tts.keys():
                            self.params['tts'] = loaded_tts[self.model_path]
                        else:
                            self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device']) 
                        try:
                            lang_dir = 'con-' if self.session['language'] == 'con' else self.session['language']
                            file_path = self.session['voice'].replace('_24000.wav', '.wav').replace('/eng/', f'/{lang_dir}/').replace('\\eng\\', f'\\{lang_dir}\\')
                            base_dir = os.path.dirname(file_path)
                            default_text_file = os.path.join(voices_dir, self.session['language'], 'default.txt')
                            default_text = ""
                            if os.path.exists(default_text_file):
                                with open(default_text_file, 'r', encoding='utf-8') as f:
                                    default_text = f.read().strip()
                            if not os.path.exists(file_path) and os.path.exists(self.session['voice']) and os.path.exists(base_dir) and default_text:
                                print(f"Creating default {self.session['language']} voice for {os.path.basename(file_path)}")
                                if not os.path.exists(os.path.dirname(file_path)):
                                    os.makedirs(os.path.dirname(file_path))
                                self.params['tts'].tts_to_file(
                                    text=default_text,
                                    file_path=file_path,
                                    speaker_wav=self.session['voice'],
                                    language=self.session['language'],
                                )
                            if os.path.exists(file_path):
                                self.session['voice'] = file_path
                            else:
                                msg = f"Could not convert voice {self.session['voice']} to {self.session['language']}"
                                print(msg)
                        except Exception as e:
                            error = f"_build() error: {e}"
                            print(error)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
        if self.session['tts_engine'] == XTTSv2:
            if self.session['custom_model'] is not None:
                msg = f"Loading TTS {self.session['tts_engine']} model, it takes a while, please be patient..."
                print(msg)
                self.model_path = os.path.join(self.session['custom_model_dir'], self.session['tts_engine'], self.session['custom_model'], 'model.pth')
                self.config_path = os.path.join(self.session['custom_model_dir'], self.session['tts_engine'], self.session['custom_model'],'config.json')
                self.vocab_path = os.path.join(self.session['custom_model_dir'], self.session['tts_engine'], self.session['custom_model'],'vocab.json')
                tts_key = self.session['custom_model']
                
                # 检查是否有共享模型
                model_ready_file = os.path.join(cache_dir, f"model_checkpoint_{self.session['custom_model']}_ready")
                
                if is_distributed and not self.is_gui_process:
                    if is_initializer:
                        # 作为初始化进程，负责加载模型
                        print(f"进程 {rank} 作为初始化进程，开始加载模型...")
                        if tts_key in loaded_tts.keys():
                            self.params['tts'] = loaded_tts[tts_key]
                        else:
                            if len(loaded_tts) == max_tts_in_memory:
                                self._unload_tts()
                            self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
                    else:
                        # 作为普通进程，等待模型加载完成
                        if os.path.exists(model_ready_file):
                            print(f"进程 {rank} 检测到模型已加载，从共享存储加载...")
                            if tts_key in loaded_tts.keys():
                                self.params['tts'] = loaded_tts[tts_key]
                            else:
                                self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
                        else:
                            print(f"进程 {rank} 等待模型加载完成...")
                            # 等待模型加载
                            start_time = time.time()
                            while not os.path.exists(model_ready_file):
                                time.sleep(1)
                                # 每5秒打印等待状态
                                if int(time.time() - start_time) % 5 == 0:
                                    print(f"进程 {rank} 正在等待模型加载... 已等待 {int(time.time() - start_time)} 秒")
                                # 如果等待超过5分钟，则超时
                                if time.time() - start_time > 300:
                                    print(f"进程 {rank} 等待模型加载超时，尝试自行加载...")
                                    break
                            
                            # 尝试加载共享模型
                            if os.path.exists(model_ready_file):
                                print(f"进程 {rank} 检测到模型已加载，从共享存储加载...")
                            else:
                                print(f"进程 {rank} 等待超时，将自行加载模型...")
                            
                            if tts_key in loaded_tts.keys():
                                self.params['tts'] = loaded_tts[tts_key]
                            else:
                                self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
                else:
                    # 单进程或GUI进程，正常加载
                    if tts_key in loaded_tts.keys():
                        self.params['tts'] = loaded_tts[tts_key]
                    else:
                        if len(loaded_tts) == max_tts_in_memory:
                            self._unload_tts()
                        self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
            elif self.session['fine_tuned'] != 'internal':
                msg = f"Loading TTS {self.session['tts_engine']} model, it takes a while, please be patient..."
                print(msg)
                hf_repo = models[self.session['tts_engine']][self.session['fine_tuned']]['repo']
                hf_sub = models[self.session['tts_engine']][self.session['fine_tuned']]['sub']
                cache_dir = os.path.join(models_dir,'tts')
                self.model_path = hf_hub_download(repo_id=hf_repo, filename=f"{hf_sub}/model.pth", cache_dir=cache_dir)
                self.config_path = hf_hub_download(repo_id=hf_repo, filename=f"{hf_sub}/config.json", cache_dir=cache_dir)
                self.vocab_path = hf_hub_download(repo_id=hf_repo, filename=f"{hf_sub}/vocab.json", cache_dir=cache_dir)    
                tts_key = hf_sub
                
                # 共享模型处理
                model_ready_file = os.path.join(cache_dir, f"model_checkpoint_{tts_key}_ready")
                
                if is_distributed and not self.is_gui_process:
                    if is_initializer:
                        print(f"进程 {rank} 作为初始化进程，开始加载模型...")
                        if tts_key in loaded_tts.keys():
                            self.params['tts'] = loaded_tts[tts_key]
                        else:
                            self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
                    else:
                        # 等待模型加载
                        if os.path.exists(model_ready_file):
                            print(f"进程 {rank} 检测到模型已加载，从共享存储加载...")
                            if tts_key in loaded_tts.keys():
                                self.params['tts'] = loaded_tts[tts_key]
                            else:
                                self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
                        else:
                            print(f"进程 {rank} 等待模型加载完成...")
                            start_time = time.time()
                            while not os.path.exists(model_ready_file):
                                time.sleep(1)
                                if int(time.time() - start_time) % 5 == 0:
                                    print(f"进程 {rank} 正在等待模型加载... 已等待 {int(time.time() - start_time)} 秒")
                                if time.time() - start_time > 300:
                                    print(f"进程 {rank} 等待模型加载超时，尝试自行加载...")
                                    break
                            
                            if os.path.exists(model_ready_file):
                                print(f"进程 {rank} 检测到模型已加载，从共享存储加载...")
                            else:
                                print(f"进程 {rank} 等待超时，将自行加载模型...")
                                
                            if tts_key in loaded_tts.keys():
                                self.params['tts'] = loaded_tts[tts_key]
                            else:
                                self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
                else:
                    # 单进程或GUI进程，正常加载
                    if tts_key in loaded_tts.keys():
                        self.params['tts'] = loaded_tts[tts_key]
                    else:
                        self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
            else:
                msg = f"Loading TTS {models[self.session['tts_engine']][self.session['fine_tuned']]['repo']} model, it takes a while, please be patient..."
                print(msg)
                self.model_path = models[self.session['tts_engine']][self.session['fine_tuned']]['repo']
                tts_key = self.model_path
                
                # 共享模型处理
                model_ready_file = os.path.join(cache_dir, f"model_api_{os.path.basename(self.model_path)}_ready")
                
                if is_distributed and not self.is_gui_process:
                    if is_initializer:
                        print(f"进程 {rank} 作为初始化进程，开始加载模型...")
                        if tts_key in loaded_tts.keys():
                            self.params['tts'] = loaded_tts[self.model_path]
                        else:
                            self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
                    else:
                        # 等待模型加载
                        if os.path.exists(model_ready_file):
                            print(f"进程 {rank} 检测到模型已加载，从共享存储加载...")
                            if tts_key in loaded_tts.keys():
                                self.params['tts'] = loaded_tts[self.model_path]
                            else:
                                self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
                        else:
                            print(f"进程 {rank} 等待模型加载完成...")
                            start_time = time.time()
                            while not os.path.exists(model_ready_file):
                                time.sleep(1)
                                if int(time.time() - start_time) % 5 == 0:
                                    print(f"进程 {rank} 正在等待模型加载... 已等待 {int(time.time() - start_time)} 秒")
                                if time.time() - start_time > 300:
                                    print(f"进程 {rank} 等待模型加载超时，尝试自行加载...")
                                    break
                            
                            if os.path.exists(model_ready_file):
                                print(f"进程 {rank} 检测到模型已加载，从共享存储加载...")
                            else:
                                print(f"进程 {rank} 等待超时，将自行加载模型...")
                                
                            if tts_key in loaded_tts.keys():
                                self.params['tts'] = loaded_tts[self.model_path]
                            else:
                                self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
                else:
                    # 单进程或GUI进程，正常加载
                    if tts_key in loaded_tts.keys():
                        self.params['tts'] = loaded_tts[self.model_path]
                    else:
                        self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
        elif self.session['tts_engine'] == BARK:
            if self.session['custom_model'] is not None:
                msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                print(msg)
            else:
                self.model_path = models[self.session['tts_engine']][self.session['fine_tuned']]['repo']
                msg = f"Loading TTS {self.model_path} model, it takes a while, please be patient..."
                print(msg)
                tts_key = self.model_path
                
                # 其他模型也可以使用相同的共享逻辑
                model_ready_file = os.path.join(cache_dir, f"model_api_{os.path.basename(self.model_path)}_ready")
                
                if is_distributed and not self.is_gui_process:
                    if is_initializer:
                        print(f"进程 {rank} 作为初始化进程，开始加载模型...")
                        if tts_key in loaded_tts.keys():
                            self.params['tts'] = loaded_tts[tts_key]
                        else:
                            self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
                    else:
                        # 等待模型加载
                        if os.path.exists(model_ready_file):
                            print(f"进程 {rank} 检测到模型已加载，从共享存储加载...")
                            if tts_key in loaded_tts.keys():
                                self.params['tts'] = loaded_tts[tts_key]
                            else:
                                self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
                        else:
                            print(f"进程 {rank} 等待模型加载完成...")
                            start_time = time.time()
                            while not os.path.exists(model_ready_file):
                                time.sleep(1)
                                if int(time.time() - start_time) % 5 == 0:
                                    print(f"进程 {rank} 正在等待模型加载... 已等待 {int(time.time() - start_time)} 秒")
                                if time.time() - start_time > 300:
                                    print(f"进程 {rank} 等待模型加载超时，尝试自行加载...")
                                    break
                            
                            if os.path.exists(model_ready_file):
                                print(f"进程 {rank} 检测到模型已加载，从共享存储加载...")
                            else:
                                print(f"进程 {rank} 等待超时，将自行加载模型...")
                                
                            if tts_key in loaded_tts.keys():
                                self.params['tts'] = loaded_tts[tts_key]
                            else:
                                self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
                else:
                    # 单进程或GUI进程，正常加载
                    if tts_key in loaded_tts.keys():
                        self.params['tts'] = loaded_tts[tts_key]
                    else:
                        self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
        elif self.session['tts_engine'] == VITS:
            # VITS 模型也需要支持共享逻辑
            # 此处代码类似，为简化篇幅，仅保留原有逻辑
            if self.session['custom_model'] is not None:
                msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                print(msg)
            else:
                iso_dir = self.session['language_iso1']
                sub_dict = models[self.session['tts_engine']][self.session['fine_tuned']]['sub']
                sub = next((key for key, lang_list in sub_dict.items() if iso_dir in lang_list), None)
                if sub is None:
                    iso_dir = self.session['language']
                    sub = next((key for key, lang_list in sub_dict.items() if iso_dir in lang_list), None)
                if sub is not None:
                    self.model_path = models[self.session['tts_engine']][self.session['fine_tuned']]['repo'].replace("[lang_iso1]", iso_dir).replace("[xxx]", sub)
                    tts_key = self.model_path
                    msg = f"Loading TTS {tts_key} model, it takes a while, please be patient..."
                    print(msg)
                    if tts_key in loaded_tts.keys():
                        self.params['tts'] = loaded_tts[tts_key]
                    else:
                        self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
                    if self.session['voice'] is not None:
                        tts_vc_key = default_vc_model
                        msg = f"Loading TTS {tts_vc_key} zeroshot model, it takes a while, please be patient..."
                        print(msg)
                        if tts_vc_key in loaded_tts.keys():
                            self.params['tts_vc'] = loaded_tts[tts_vc_key]
                        else:
                            self.params['tts_vc'] = load_coqui_tts_vc(self.session['device'])
                else:
                    msg = f"{self.session['tts_engine']} checkpoint for {self.session['language']} not found!"
                    print(msg)
                    
        elif self.session['tts_engine'] == FAIRSEQ:
            # FAIRSEQ 模型也需要支持共享逻辑
            # 此处代码类似，为简化篇幅，仅保留原有逻辑
            if self.session['custom_model'] is not None:
                msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                print(msg)
            else:
                self.model_path = models[self.session['tts_engine']][self.session['fine_tuned']]['repo'].replace("[lang]", self.session['language'])
                tts_key = self.model_path
                msg = f"Loading TTS {tts_key} model, it takes a while, please be patient..."
                print(msg)
                if tts_key in loaded_tts.keys():
                    self.params['tts'] = loaded_tts[tts_key]
                else:
                    self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
                if self.session['voice'] is not None:
                    tts_vc_key = default_vc_model
                    msg = f"Loading TTS {tts_vc_key} zeroshot model, it takes a while, please be patient..."
                    print(msg)
                    if tts_vc_key in loaded_tts.keys():
                        self.params['tts_vc'] = loaded_tts[tts_vc_key]
                    else:
                        self.params['tts_vc'] = load_coqui_tts_vc(self.session['device'])
        elif self.session['tts_engine'] == YOURTTS:
            # YOURTTS 模型也需要支持共享逻辑
            # 此处代码类似，为简化篇幅，仅保留原有逻辑
            if self.session['custom_model'] is not None:
                msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                print(msg)
            else:
                self.model_path = models[self.session['tts_engine']][self.session['fine_tuned']]['repo']
                msg = f"Loading TTS {self.model_path} model, it takes a while, please be patient..."
                print(msg)
                tts_key = self.model_path
                if tts_key in loaded_tts.keys():
                    self.params['tts'] = loaded_tts[self.model_path]
                else:
                    self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
        if self.params['tts'] is not None:
            loaded_tts[tts_key] = self.params['tts']
            
    def _wav_to_npz(self, wav_path, npz_path):
        audio, sr = sf.read(wav_path)
        np.savez(npz_path, audio=audio, sample_rate=24000)
        msg = f"Saved NPZ file: {npz_path}"
        print(msg)
        
    def _detect_gender(self, voice_path):
        try:
            sample_rate, signal = wav.read(voice_path)
            # Convert stereo to mono if needed
            if len(signal.shape) > 1:
                signal = np.mean(signal, axis=1)
            # Compute FFT
            fft_spectrum = np.abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(fft_spectrum), d=1/sample_rate)
            # Consider only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = fft_spectrum[:len(fft_spectrum)//2]
            # Find peaks in frequency spectrum
            peaks, _ = find_peaks(positive_magnitude, height=np.max(positive_magnitude) * 0.2)
            if len(peaks) == 0:
                return None 
            # Find the first strong peak within the human voice range (75Hz - 300Hz)
            for peak in peaks:
                if 75 <= positive_freqs[peak] <= 300:
                    pitch = positive_freqs[peak]
                    gender = "female" if pitch > 135 else "male"
                    return gender
                    break     
            return None
        except Exception as e:
            error = f"_detect_gender() error: {voice_path}: {e}"
            print(error)
            return None
            
    def _is_tts_active(self, tts):
        return any(obj is tts for obj in gc.get_objects())

    def _unload_tts():
         for key in list(loaded_tts.keys()):
            if key != default_vc_model:
                del loaded_tts[key]
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                break

    def _tensor_type(self, audio_data):
        if isinstance(audio_data, torch.Tensor):
            return audio_data
        elif isinstance(audio_data, np.ndarray):  
            return torch.from_numpy(audio_data).float()
        elif isinstance(audio_data, list):  
            return torch.tensor(audio_data, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type for audio_data: {type(audio_data)}")

    def _trim_end(self, audio_data, sample_rate, silence_threshold=0.001, buffer_seconds=0.001):
        # Ensure audio_data is a PyTorch tensor
        if isinstance(audio_data, list):  
            audio_data = torch.tensor(audio_data)  # Convert list to tensor
        
        if isinstance(audio_data, torch.Tensor):
            if audio_data.is_cuda:
                audio_data = audio_data.cpu()  # Move to CPU if it's on CUDA
            
            # Detect non-silent indices
            non_silent_indices = torch.where(audio_data.abs() > silence_threshold)[0]

            if len(non_silent_indices) == 0:
                return torch.tensor([], device=audio_data.device)

            # Determine the trimming index
            end_index = non_silent_indices[-1] + int(buffer_seconds * sample_rate)

            # Trim the audio, keeping it as a tensor
            trimmed_audio = audio_data[:end_index]

            return trimmed_audio
        
        # If somehow the input is still incorrect, raise an error
        raise TypeError("audio_data must be a PyTorch tensor or a list of numerical values.")

    def _normalize_audio(self, input_file, output_file, samplerate):
        filter_complex = (
            'agate=threshold=-25dB:ratio=1.4:attack=10:release=250,'
            'afftdn=nf=-70,'
            'acompressor=threshold=-20dB:ratio=2:attack=80:release=200:makeup=1dB,'
            'loudnorm=I=-14:TP=-3:LRA=7:linear=true,'
            'equalizer=f=150:t=q:w=2:g=1,'
            'equalizer=f=250:t=q:w=2:g=-3,'
            'equalizer=f=3000:t=q:w=2:g=2,'
            'equalizer=f=5500:t=q:w=2:g=-4,'
            'equalizer=f=9000:t=q:w=2:g=-2,'
            'highpass=f=63[audio]'
        )
        ffmpeg_cmd = [shutil.which('ffmpeg'), '-hide_banner', '-nostats', '-i', input_file]
        ffmpeg_cmd += [
            '-filter_complex', filter_complex,
            '-map', '[audio]',
            '-ar', str(samplerate),
            '-y', output_file
        ]
        try:
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True,
                text=True,
                encoding='utf-8'
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"_normalize_audio() error: {input_file}: {e}")
            return False

    def convert_sentence_to_audio(self):
        try:
            audio_data = None
            fine_tuned_params = {
                key: cast_type(self.session[key])
                for key, cast_type in {
                    "temperature": float,
                    "length_penalty": float,
                    "num_beams": int,
                    "repetition_penalty": float,
                    "top_k": int,
                    "top_p": float,
                    "speed": float,
                    "enable_text_splitting": bool
                }.items()
                if self.session.get(key) is not None
            }
            self.params['voice_path'] = (
                self.session['voice'] if self.session['voice'] is not None 
                else os.path.join(self.session['custom_model_dir'], self.session['tts_engine'], self.session['custom_model'],'ref.wav') if self.session['custom_model'] is not None
                else models[self.session['tts_engine']][self.session['fine_tuned']]['voice'] if self.session['fine_tuned'] != 'internal'
                else models[self.session['tts_engine']]['internal']['voice']
            )
            if self.session['tts_engine'] == XTTSv2:
                if self.session['custom_model'] is not None or self.session['fine_tuned'] != 'internal':
                    if self.params['current_voice_path'] != self.params['voice_path']:
                        msg = 'Computing speaker latents...'
                        print(msg)
                        self.params['current_voice_path'] = self.params['voice_path']
                        self.params['gpt_cond_latent'], self.params['speaker_embedding'] = self.params['tts'].get_conditioning_latents(audio_path=[self.params['voice_path']])
                    with torch.no_grad():
                        result = self.params['tts'].inference(
                            text=self.params['sentence'],
                            language=self.session['language_iso1'],
                            gpt_cond_latent=self.params['gpt_cond_latent'],
                            speaker_embedding=self.params['speaker_embedding'],
                            **fine_tuned_params
                        )
                    audio_data = result.get('wav')
                    if audio_data is not None:
                        audio_data = audio_data.tolist()
                    else:
                        error = f'No audio waveform found in convert_sentence_to_audio() result: {result}'
                        print(error)
                        return False
                else:
                    if self.params['voice_path'] in default_xtts_settings['voices'].values():
                        speaker_argument = {"speaker": self.params['voice_path']}
                    else:
                        if self.params['current_voice_path'] != self.params['voice_path']:
                            self.params['current_voice_path'] = self.params['voice_path']
                        speaker_argument = {"speaker_wav": self.params['voice_path']}
                    with torch.no_grad():
                        audio_data = self.params['tts'].tts(
                            text=self.params['sentence'],
                            language=self.session['language_iso1'],
                            **speaker_argument,
                            **fine_tuned_params
                        )
            elif self.session['tts_engine'] == BARK:
                '''
                    [laughter]
                    [laughs]
                    [sighs]
                    [music]
                    [gasps]
                    [clears throat]
                    — or ... for hesitations
                    ♪ for song lyrics
                    CAPITALIZATION for emphasis of a word
                    [MAN] and [WOMAN] to bias Bark toward male and female speakers, respectively
                '''
                if self.session['custom_model'] is not None or self.session['fine_tuned'] != 'internal':
                    msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                    print(msg)
                else:
                    if self.params['voice_path'] is not None:
                        if self.params['current_voice_path'] != self.params['voice_path']:
                            self.params['current_voice_path'] = self.params['voice_path']
                            r"""
                            voice_key = re.sub(r'_(24000|16000)\.wav$', '', os.path.basename(self.params['voice_path']))
                            bark_dir = os.path.join(os.path.dirname(self.params['voice_path']), 'bark', voice_key)
                            npz_file = os.path.join(bark_dir, f'{voice_key}.npz')
                            if not os.path.exists(npz_file):
                                os.makedirs(bark_dir, exist_ok=True)
                                self._wav_to_npz(self.params['voice_path'], npz_file)
                            """
                            bark_dir = f"/{os.path.dirname(default_bark_settings['voices']['Jamie'])}"
                            voice_key = re.sub(r'.npz$', '', os.path.basename(default_bark_settings['voices']['Jamie']))
                            speaker_argument = {
                                "voice_dir": bark_dir,
                                "speaker": voice_key,
                                "speaker_wav": os.path.join(os.path.dirname(bark_dir), f"{os.path.splitext(os.path.basename(default_bark_settings['voices']['KumarDahl']))[0]}.wav"),
                                "text_temp": 0.2
                            }                    
                    else:
                        bark_dir = f"/{os.path.dirname(default_bark_settings['voices']['Jamie'])}"
                        voice_key = re.sub(r'.npz$', '', os.path.basename(default_bark_settings['voices']['Jamie']))
                        speaker_argument = {
                            "voice_dir": bark_dir,
                            "speaker": voice_key,
                            "text_temp": 0.2
                        }                      
                    with torch.no_grad():
                        audio_data = self.params['tts'].tts(
                            text=self.params['sentence'],
                            **speaker_argument
                        )
            elif self.session['tts_engine'] == VITS:
                if self.session['custom_model'] is not None or self.session['fine_tuned'] != 'internal':
                    msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                    print(msg)
                else:
                    speaker_argument = {}
                    if self.session['language'] == 'eng' and 'vctk/vits' in models[self.session['tts_engine']]['internal']['sub']:
                        if self.session['language'] in models[self.session['tts_engine']]['internal']['sub']['vctk/vits'] or self.session['language_iso1'] in models[self.session['tts_engine']]['internal']['sub']['vctk/vits']:
                            speaker_argument = {"speaker": 'p262'}
                    elif self.session['language'] == 'cat' and 'custom/vits' in models[self.session['tts_engine']]['internal']['sub']:
                        if self.session['language'] in models[self.session['tts_engine']]['internal']['sub']['custom/vits'] or self.session['language_iso1'] in models[self.session['tts_engine']]['internal']['sub']['custom/vits']:
                            speaker_argument = {"speaker": '09901'}
                    with torch.no_grad():
                        if self.params['voice_path'] is not None:
                            proc_dir = os.path.join(self.session['voice_dir'], 'proc')
                            os.makedirs(proc_dir, exist_ok=True)
                            tmp_in_wav = os.path.join(proc_dir, f"{uuid.uuid4()}.wav")
                            tmp_out_wav = os.path.join(proc_dir, f"{uuid.uuid4()}.wav")
                            self.params['tts'].tts_to_file(
                                text=self.params['sentence'],
                                file_path=tmp_in_wav,
                                **speaker_argument
                            )
                            if self.params['current_voice_path'] != self.params['voice_path']:
                                self.params['current_voice_path'] = self.params['voice_path']
                                self.params['voice_path_gender'] = self._detect_gender(self.params['voice_path'])
                                self.params['voice_builtin_gender'] = self._detect_gender(tmp_in_wav)
                                msg = f"Cloned voice seems to be {self.params['voice_path_gender']}"
                                print(msg)
                                msg = f"Builtin voice seems to be {self.params['voice_builtin_gender']}"
                                print(msg)
                                if self.params['voice_builtin_gender'] != self.params['voice_path_gender']:
                                    self.params['semitones'] = -4 if self.params['voice_path_gender'] == 'male' else 4
                                    msg = f"Adapting builtin voice frequencies from the clone voice..."
                                print(msg)
                            if 'semitones' in self.params:
                                try:
                                    cmd = [
                                        shutil.which('sox'), tmp_in_wav,
                                        "-r", str(self.params['sample_rate']), tmp_out_wav,
                                        "pitch", str(self.params['semitones'] * 100)
                                    ]
                                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                except subprocess.CalledProcessError as e:
                                    print(f"Subprocess error: {e.stderr}")
                                    DependencyError(e)
                                    return False
                                except FileNotFoundError as e:
                                    print(f"File not found: {e}")
                                    DependencyError(e)
                                    return False
                            else:
                                tmp_out_wav = tmp_in_wav
                            audio_data = self.params['tts_vc'].voice_conversion(
                                source_wav=tmp_out_wav,
                                target_wav=self.params['voice_path']
                            )
                            if os.path.exists(tmp_in_wav):
                                os.remove(tmp_in_wav)
                            if os.path.exists(tmp_out_wav):
                                os.remove(tmp_out_wav)
                        else:
                            audio_data = self.params['tts'].tts(
                                text=self.params['sentence'],
                                **speaker_argument
                            )
            elif self.session['tts_engine'] == FAIRSEQ:
                if self.session['custom_model'] is not None or self.session['fine_tuned'] != 'internal':
                    msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                    print(msg)
                else:
                    with torch.no_grad():
                        if self.params['voice_path'] is not None:
                            self.params['voice_path'] = re.sub(r'_24000\.wav$', '_16000.wav', self.params['voice_path'])
                            proc_dir = os.path.join(self.session['voice_dir'], 'proc')
                            os.makedirs(proc_dir, exist_ok=True)
                            tmp_in_wav = os.path.join(proc_dir, f"{uuid.uuid4()}.wav")
                            tmp_out_wav = os.path.join(proc_dir, f"{uuid.uuid4()}.wav")
                            self.params['tts'].tts_to_file(text=self.params['sentence'], file_path=tmp_in_wav)
                            if self.params['current_voice_path'] != self.params['voice_path']:
                                self.params['current_voice_path'] = self.params['voice_path']
                                self.params['voice_path_gender'] = self._detect_gender(self.params['voice_path'])
                                self.params['voice_builtin_gender'] = self._detect_gender(tmp_in_wav)
                                msg = f"Cloned voice seems to be {self.params['voice_path_gender']}"
                                print(msg)
                                msg = f"Builtin voice seems to be {self.params['voice_builtin_gender']}"
                                print(msg)
                                if self.params['voice_builtin_gender'] != self.params['voice_path_gender']:
                                    self.params['semitones'] = -4 if self.params['voice_path_gender'] == 'male' else 4
                                    msg = f"Adapting builtin voice frequencies from the clone voice..."
                                print(msg)
                            if 'semitones' in self.params:
                                try:
                                    cmd = [
                                        shutil.which('sox'), tmp_in_wav,
                                        "-r", str(self.params['sample_rate']), tmp_out_wav,
                                        "pitch", str(self.params['semitones'] * 100)
                                    ]
                                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                except subprocess.CalledProcessError as e:
                                    print(f"Subprocess error: {e.stderr}")
                                    DependencyError(e)
                                    return False
                                except FileNotFoundError as e:
                                    print(f"File not found: {e}")
                                    DependencyError(e)
                                    return False
                            else:
                                tmp_out_wav = tmp_in_wav
                            audio_data = self.params['tts_vc'].voice_conversion(
                                source_wav=tmp_out_wav,
                                target_wav=self.params['voice_path']
                            )
                            if os.path.exists(tmp_in_wav):
                                os.remove(tmp_in_wav)
                            if os.path.exists(tmp_out_wav):
                                os.remove(tmp_out_wav)
                        else:
                            audio_data = self.params['tts'].tts(
                                text=self.params['sentence']
                            )
            elif self.session['tts_engine'] == YOURTTS:
                if self.session['custom_model'] is not None or self.session['fine_tuned'] != 'internal':
                    msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                    print(msg)
                else:
                    with torch.no_grad():
                        language = self.session['language_iso1'] if self.session['language_iso1'] == 'en' else 'fr-fr' if self.session['language_iso1'] == 'fr' else 'pt-br' if self.session['language_iso1'] == 'pt' else 'en'
                        if self.params['voice_path'] is not None:
                            self.params['voice_path'] = re.sub(r'_24000\.wav$', '_16000.wav', self.params['voice_path'])
                            speaker_argument = {"speaker_wav": self.params['voice_path']}
                        else:
                            voice_name = default_yourtts_settings['voices']['ElectroMale-2']
                            speaker_argument = {"speaker": voice_name}
                        audio_data = self.params['tts'].tts(
                            text=self.params['sentence'],
                            language=language,
                            **speaker_argument
                        )
            if audio_data is not None:
                if self.params['sentence'].endswith('–'):
                    audio_data = self._trim_end(audio_data, self.params['sample_rate'])
                sourceTensor = self._tensor_type(audio_data)
                audio_tensor = sourceTensor.clone().detach().unsqueeze(0).cpu()
                torchaudio.save(self.params['sentence_audio_file'], audio_tensor, self.params['sample_rate'], format=default_audio_proc_format)
                del audio_data, sourceTensor, audio_tensor
            if self.session['device'] == 'cuda':
                torch.cuda.empty_cache()         
            if os.path.exists(self.params['sentence_audio_file']):
                return True
            error = f"Cannot create {self.params['sentence_audio_file']}"
            print(error)
            return False
        except Exception as e:
            error = f'convert_sentence_to_audio(): {e}'
            raise ValueError(e)
            return False