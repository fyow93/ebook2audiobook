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
from torch.nn.parallel import DistributedDataParallel as DDP

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

@app.post("/load_coqui_tts_api/")
def load_coqui_tts_api(model_path, device):
    try:
        with lock:
            tts = TtsXTTS(model_path)
            if device == 'cuda':
                tts.cuda()
            else:
                tts.to(device)
        return tts
    except Exception as e:
        error = f'load_coqui_tts_api() error: {e}'
        print(error)
        return None

@app.post("/load_coqui_tts_checkpoint/")
def load_coqui_tts_checkpoint(model_path, config_path, vocab_path, device):
    try:
        # 设置环境变量确保使用所有GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 使用所有4个GPU
        
        # 获取当前GPU ID (如果是分布式模式)
        gpu_id = int(os.environ.get('LOCAL_RANK', '0'))
        print(f"当前进程使用GPU ID: {gpu_id}")
        
        # 检测可用GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU设备")
        
        # 判断是否为分布式模式
        is_distributed = os.environ.get('RANK') is not None and os.environ.get('WORLD_SIZE') is not None
        if is_distributed:
            print(f"运行在分布式模式下: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
        
        # 初始化DeepSpeed的分布式环境
        if num_gpus > 1 and default_xtts_settings['use_deepspeed']:
            # 尝试初始化分布式环境
            try:
                if not dist.is_initialized() and is_distributed:
                    # 使用当前进程的分布式环境
                    dist_backend = "nccl"  # GPU推荐使用nccl
                    print(f"初始化分布式进程组: {dist_backend}")
                    dist.init_process_group(backend=dist_backend)
                    print(f"分布式环境已初始化")
            except Exception as e:
                print(f"初始化分布式环境失败: {e}")
                print("回退到单GPU模式")
                
        # 明确设置当前设备
        if device == 'cuda':
            # 在分布式模式下，使用分配的GPU
            if is_distributed:
                torch.cuda.set_device(gpu_id)
                current_device = torch.cuda.current_device()
                print(f"已设置为GPU {current_device}")
            else:
                # 非分布式模式下使用GPU 0
                torch.cuda.set_device(0)
        
        # 加载配置
        config = XttsConfig()
        config.models_dir = os.path.join("models", "tts")
        config.load_json(config_path)
        
        # 加载DeepSpeed配置
        ds_config = None
        if default_xtts_settings['use_deepspeed'] and num_gpus > 1:
            try:
                # 总是尝试从ds_config.json加载配置
                config_path = "ds_config.json"
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        ds_config = json.load(f)
                    print(f"已加载DeepSpeed配置文件: {config_path}")
                    
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
                    print(f"警告: 未找到DeepSpeed配置文件 {config_path}")
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
        
        with lock:
            tts.load_checkpoint(
                config,
                checkpoint_path=model_path,
                vocab_path=vocab_path,
                use_deepspeed=default_xtts_settings['use_deepspeed'] and num_gpus > 1,
                eval=True,
                deepspeed_config=ds_config
            )
        
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
            
        return tts
    except Exception as e:
        error = f'load_coqui_tts_checkpoint() error: {e}'
        print(error)
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
        if self.session['language'] in language_tts[XTTSv2].keys():
            if self.session['voice'] is not None and self.session['language'] != 'eng':
                voice_key = re.sub(r'_(24000|16000)\.wav$', '', os.path.basename(self.session['voice']))
                if voice_key in default_xtts_settings['voices']:
                    if not f"/{self.session['language']}/" in self.session['voice']:
                        msg = f"Converting xttsv2 builtin english voice to {self.session['language']}..."
                        print(msg)
                        self.model_path = models[XTTSv2]['internal']['repo']
                        tts_key = self.model_path
                        if tts_key in loaded_tts.keys():
                            self.params['tts'] = loaded_tts[self.model_path]
                        else:
                            self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device']) 
                        try:
                            lang_dir = 'con-' if self.session['language'] == 'con' else self.session['language']
                            file_path = self.session['voice'].replace('_24000.wav', '.wav').replace('/eng/', f'/{lang_dir}/').replace('\\eng\\', f'\\{lang_dir}\\')
                            base_dir = os.path.dirname(file_path)
                            default_text_file = os.path.join(voices_dir, self.session['language'], 'default.txt')
                            if os.path.exists(default_text_file):
                                default_text = Path(default_text_file).read_text(encoding="utf-8")
                                self.params['tts'].tts_to_file(
                                    text=f"{default_xtts_settings['voices'][voice_key]}, {default_text}",
                                    speaker=default_xtts_settings['voices'][voice_key],
                                    language=self.session['language_iso1'],
                                    file_path=file_path
                                )
                                for samplerate in [16000, 24000]:
                                    output_file = file_path.replace('.wav', f'_{samplerate}.wav')
                                    if self._normalize_audio(file_path, output_file, samplerate):
                                        # for Bark
                                        if samplerate == 24000:
                                            bark_dir = os.path.join(base_dir, 'bark')
                                            npz_dir = os.path.join(bark_dir, voice_key)
                                            os.makedirs(npz_dir, exist_ok=True)
                                            npz_file = os.path.join(npz_dir, f'{voice_key}.npz')
                                            self._wav_to_npz(output_file, npz_file)
                                    else:
                                        break
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                            else:
                                error = f'The translated {default_text_file} could not be found! Voice cloning file will stay in English.'
                                print(error)
                        except Exception as e:
                            error = f'_build() builtin voice conversion error: {file_path}: {e}'
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
                if self.session['custom_model'] in loaded_tts.keys():
                    self.params['tts'] = loaded_tts[self.session['custom_model']]
                else:
                    if len(loaded_tts) == max_tts_in_memory:
                        self.unload_tts()
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
                if tts_key in loaded_tts.keys():
                    self.params['tts'] = loaded_tts[hf_sub]
                else:
                    self.params['tts'] = load_coqui_tts_checkpoint(self.model_path, self.config_path, self.vocab_path, self.session['device'])
            else:
                msg = f"Loading TTS {models[self.session['tts_engine']][self.session['fine_tuned']]['repo']} model, it takes a while, please be patient..."
                print(msg)
                self.model_path = models[self.session['tts_engine']][self.session['fine_tuned']]['repo']
                tts_key = self.model_path
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
                if tts_key in loaded_tts.keys():
                    self.params['tts'] = loaded_tts[tts_key]
                else:
                    self.params['tts'] = load_coqui_tts_api(self.model_path, self.session['device'])
        elif self.session['tts_engine'] == VITS:
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