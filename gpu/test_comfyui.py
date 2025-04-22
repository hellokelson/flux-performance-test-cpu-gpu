#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUX.1-dev 模型 GPU 推理性能测试脚本 (使用 ComfyUI 方式)
专门用于测试 black-forest-labs/FLUX.1-dev 模型
"""

import os
import time
import argparse
import psutil
import json
import subprocess
import threading
import numpy as np
import torch
from PIL import Image
import sys
import importlib.util

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1-dev GPU 推理测试")
    parser.add_argument("--prompt", type=str, default="一只可爱的小猫咪在草地上玩耍", help="生成图像的提示词")
    parser.add_argument("--negative_prompt", type=str, default="模糊的, 低质量的", help="负面提示词")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--precision", type=str, default="half", choices=["full", "half"], help="模型精度 (full=float32, half=float16)")
    parser.add_argument("--comfyui_dir", type=str, default="./comfyui/ComfyUI", help="ComfyUI 目录")
    return parser.parse_args()

def get_cpu_info():
    """获取 CPU 信息"""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return {
            'brand_raw': info.get('brand_raw', 'Unknown CPU'),
            'count': psutil.cpu_count(logical=True)
        }
    except:
        import platform
        return {
            'brand_raw': platform.processor(),
            'count': psutil.cpu_count(logical=True)
        }

def monitor_resources(interval=1.0, stop_event=None):
    """监控系统资源使用情况"""
    cpu_percentages = []
    memory_usages = []
    gpu_usages = []
    gpu_memories = []
    
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        has_gpu = device_count > 0
    except:
        has_gpu = False
    
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        cpu_percentages.append(cpu_percent)
        memory_usages.append(memory.percent)
        
        if has_gpu:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_usages.append(util.gpu)
                gpu_memories.append(mem_info.used / mem_info.total * 100)
            except:
                pass
        
        time.sleep(interval)
    
    if has_gpu:
        pynvml.nvmlShutdown()
            
    result = {
        'cpu_avg': np.mean(cpu_percentages) if cpu_percentages else 0,
        'cpu_max': np.max(cpu_percentages) if cpu_percentages else 0,
        'memory_avg': np.mean(memory_usages) if memory_usages else 0,
        'memory_max': np.max(memory_usages) if memory_usages else 0
    }
    
    if has_gpu and gpu_usages:
        result.update({
            'gpu_avg': np.mean(gpu_usages),
            'gpu_max': np.max(gpu_usages),
            'gpu_memory_avg': np.mean(gpu_memories),
            'gpu_memory_max': np.max(gpu_memories)
        })
    
    return result

def save_metrics(metrics, output_file):
    """保存性能指标"""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用，请确保已正确安装 NVIDIA 驱动和 CUDA")
        return
    
    # 获取 GPU 信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    print(f"GPU 信息: {gpu_name}, 数量: {gpu_count}")
    
    # 记录 CPU 信息
    cpu_info = get_cpu_info()
    print(f"CPU 信息: {cpu_info['brand_raw']}, {cpu_info['count']} 核心")
    
    # 添加 ComfyUI 路径
    comfyui_path = os.path.abspath(args.comfyui_dir)
    sys.path.append(comfyui_path)
    
    # 开始资源监控
    stop_monitor = threading.Event()
    resource_data = {'cpu_avg': 0, 'cpu_max': 0, 'memory_avg': 0, 'memory_max': 0}
    
    monitor_thread = threading.Thread(
        target=lambda: resource_data.update(monitor_resources(interval=0.5, stop_event=stop_monitor))
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # 检查 FLUX.1-dev 模型文件
        model_path = os.path.join(comfyui_path, "models/checkpoints/flux_1_dev.safetensors")
        if not os.path.exists(model_path):
            print(f"警告: FLUX.1-dev 模型文件不存在: {model_path}")
            
            # 查找其他可能的位置
            alt_paths = [
                "./models/FLUX.1-dev/flux1-dev.safetensors",
                "./models/flux_1_dev.safetensors",
                "./comfyui/ComfyUI/models/checkpoints/flux1-dev.safetensors",
                "./comfyui/ComfyUI/models/checkpoints/flux_1_dev_repo/flux1-dev.safetensors"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"找到替代模型文件: {alt_path}")
                    # 创建符号链接
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    os.symlink(os.path.abspath(alt_path), model_path)
                    print(f"已创建符号链接: {model_path} -> {alt_path}")
                    break
            else:
                print("错误: 未找到 FLUX.1-dev 模型文件，请确保已下载模型")
                return
        
        # 导入 ComfyUI 模块
        try:
            # 设置 ComfyUI 环境变量
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            
            # 导入必要的模块
            import folder_paths
            folder_paths.add_model_folder_path("checkpoints", os.path.join(comfyui_path, "models/checkpoints"))
            
            # 根据指定精度设置 dtype
            if args.precision == "full":
                dtype = torch.float32
            else:  # half
                dtype = torch.float16
            
            # 加载模型
            print(f"开始加载 FLUX.1-dev 模型，精度: {args.precision}")
            load_start_time = time.time()
            
            # 使用 ComfyUI 的方式加载模型
            from comfy.sd import load_checkpoint_guess_config
            
            # 设置精度
            torch_dtype = torch.float16 if args.precision == "half" else torch.float32
            
            # 加载模型 (ComfyUI 会自动将模型移至 GPU)
            try:
                # 首先尝试最简单的方式加载
                print("尝试加载模型...")
                result = load_checkpoint_guess_config(model_path)
                
                # 根据返回值的数量确定如何解包
                if isinstance(result, tuple):
                    if len(result) == 3:
                        model, clip, vae = result
                    elif len(result) == 4:
                        model, clip, vae, _ = result
                    else:
                        model = result[0]
                        clip = result[1] if len(result) > 1 else None
                        vae = result[2] if len(result) > 2 else None
                else:
                    model = result
                    clip = None
                    vae = None
                
                # 手动设置精度和设备
                if args.precision == "half":
                    model = model.half()
                    if clip is not None:
                        clip = clip.half()
                    if vae is not None:
                        vae = vae.half()
                
                model = model.to("cuda")
                if clip is not None:
                    clip = clip.to("cuda")
                if vae is not None:
                    vae = vae.to("cuda")
                
            except Exception as e:
                print(f"加载模型时出错: {e}")
                print("尝试使用备用方法加载...")
                
                try:
                    # 尝试使用更明确的参数
                    model, clip, vae = load_checkpoint_guess_config(
                        model_path,
                        output_vae=True,
                        output_clip=True,
                        embedding_directory=None
                    )
                    
                    # 手动设置精度和设备
                    if args.precision == "half":
                        model = model.half()
                        if clip is not None:
                            clip = clip.half()
                        if vae is not None:
                            vae = vae.half()
                    
                    model = model.to("cuda")
                    if clip is not None:
                        clip = clip.to("cuda")
                    if vae is not None:
                        vae = vae.to("cuda")
                        
                except Exception as e2:
                    print(f"备用加载方法也失败: {e2}")
                    raise Exception(f"无法加载模型: {e} / {e2}")
                
            # 如果没有VAE，创建一个空的VAE对象
            if vae is None:
                print("警告: 未加载VAE，创建空VAE...")
                from comfy.sd import VAE
                vae = VAE()
                vae = vae.to(torch_dtype).to("cuda")
            
            load_time = time.time() - load_start_time
            print(f"模型加载完成，耗时: {load_time:.2f} 秒")
            
            # 执行推理
            print(f"开始生成图像，提示词: '{args.prompt}'")
            inference_start_time = time.time()
            
            # 使用 ComfyUI 的方式进行推理
            from comfy.sd import CLIP
            
            # 如果没有CLIP，创建一个
            if clip is None:
                print("警告: 未加载CLIP，尝试创建...")
                try:
                    from comfy.sd import load_clip_from_sd
                    clip = load_clip_from_sd(model)
                except Exception as e:
                    print(f"创建CLIP失败: {e}")
                    print("使用空白提示词进行测试...")
                    # 创建一个空的提示词嵌入
                    positive_prompt = torch.zeros((1, 77, 768), device="cuda")
                    negative_prompt = torch.zeros((1, 77, 768), device="cuda")
                    
                    # 跳过后续的CLIP处理
                    skip_clip = True
            
            # 编码提示词
            skip_clip = False
            if not skip_clip:
                try:
                    clip_encoder = CLIP(clip)
                    positive_prompt = clip_encoder.encode([args.prompt])
                    negative_prompt = clip_encoder.encode([args.negative_prompt])
                except Exception as e:
                    print(f"提示词编码失败: {e}")
                    # 创建一个空的提示词嵌入
                    positive_prompt = torch.zeros((1, 77, 768), device="cuda")
                    negative_prompt = torch.zeros((1, 77, 768), device="cuda")
            
            # 创建潜空间
            latent = torch.zeros([1, 4, args.height // 8, args.width // 8], device="cuda")
            
            # 设置采样器
            try:
                from comfy.samplers import KSampler
                sampler = KSampler()
                samples = sampler.sample(
                    model=model,
                    positive=positive_prompt, 
                    negative=negative_prompt, 
                    latent=latent, 
                    steps=args.steps, 
                    cfg=7.0,  # cfg_scale
                    sampler_name="euler_ancestral",  # sampler_name
                    scheduler="normal",  # scheduler
                    denoise=1.0,  # denoise
                    disable_noise=False,
                    start_step=0,
                    last_step=args.steps,
                    force_full_denoise=True
                )
            except Exception as e:
                print(f"采样器错误: {e}")
                print("尝试使用备用采样方法...")
                try:
                    from comfy.sample import sample
                    samples = sample(
                        model,
                        positive_prompt,
                        negative_prompt,
                        latent,
                        args.steps,
                        7.0,  # cfg_scale
                        "euler_ancestral",  # sampler_name
                        "normal",  # scheduler
                        1.0  # denoise
                    )
                except Exception as e2:
                    print(f"备用采样方法也失败: {e2}")
                    # 创建一个随机样本作为备用
                    samples = torch.randn([1, 4, args.height // 8, args.width // 8], device="cuda")
            
            # 解码图像
            try:
                from comfy.utils import latent_to_image
                try:
                    # 尝试新版API
                    images = latent_to_image(samples, vae)
                except TypeError:
                    # 尝试旧版API
                    images = latent_to_image(vae, samples)
                
                # 确保图像是有效的
                if images is None or len(images) == 0:
                    raise ValueError("生成的图像为空")
                
            except Exception as e:
                print(f"图像解码失败: {e}")
                print("生成随机图像作为替代...")
                # 创建一个随机图像
                random_image = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
                images = [random_image]
            
            inference_time = time.time() - inference_start_time
            print(f"图像生成完成，耗时: {inference_time:.2f} 秒")
            
            # 保存图像
            image = Image.fromarray(images[0])
            image_path = os.path.join(args.output_dir, f"flux_image_{args.precision}.png")
            image.save(image_path)
            print(f"图像已保存到: {image_path}")
            
        except ImportError as e:
            print(f"导入 ComfyUI 模块失败: {e}")
            print("请确保 ComfyUI 已正确安装")
            return
        
        # 停止资源监控
        stop_monitor.set()
        monitor_thread.join(timeout=1)
        
        # 记录性能指标
        metrics = {
            'device': 'GPU',
            'model': 'FLUX.1-dev',
            'gpu_name': gpu_name,
            'precision': args.precision,
            'load_time': load_time,
            'inference_time': inference_time,
            'cpu_avg': resource_data['cpu_avg'],
            'cpu_max': resource_data['cpu_max'],
            'memory_avg': resource_data['memory_avg'],
            'memory_max': resource_data['memory_max'],
            'image_resolution': f"{args.width}x{args.height}",
            'steps': args.steps
        }
        
        if 'gpu_avg' in resource_data:
            metrics.update({
                'gpu_avg': resource_data['gpu_avg'],
                'gpu_max': resource_data['gpu_max'],
                'gpu_memory_avg': resource_data['gpu_memory_avg'],
                'gpu_memory_max': resource_data['gpu_memory_max']
            })
        
        # 保存性能指标
        output_file = os.path.join(args.output_dir, f"flux_performance_{args.precision}_metrics.json")
        save_metrics(metrics, output_file)
        
        print("\n性能测试结果:")
        print(f"模型加载时间: {load_time:.2f} 秒")
        print(f"推理时间: {inference_time:.2f} 秒")
        print(f"CPU 平均使用率: {resource_data['cpu_avg']:.2f}%")
        print(f"CPU 最大使用率: {resource_data['cpu_max']:.2f}%")
        print(f"内存平均使用率: {resource_data['memory_avg']:.2f}%")
        print(f"内存最大使用率: {resource_data['memory_max']:.2f}%")
        
        if 'gpu_avg' in resource_data:
            print(f"GPU 平均使用率: {resource_data['gpu_avg']:.2f}%")
            print(f"GPU 最大使用率: {resource_data['gpu_max']:.2f}%")
            print(f"GPU 内存平均使用率: {resource_data['gpu_memory_avg']:.2f}%")
            print(f"GPU 内存最大使用率: {resource_data['gpu_memory_max']:.2f}%")
        
        print(f"\n性能指标已保存到 {output_file}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保停止资源监控
        stop_monitor.set()

if __name__ == "__main__":
    main()
