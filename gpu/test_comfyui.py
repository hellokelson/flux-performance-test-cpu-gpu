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
                "./comfyui/ComfyUI/models/checkpoints/flux1-dev.safetensors"
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
            model, clip, vae = load_checkpoint_guess_config(
                model_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=None,
                output_pooled=True,
                dtype=dtype
            )
            
            # 将模型移至 GPU
            model.to("cuda")
            clip.to("cuda")
            vae.to("cuda")
            
            load_time = time.time() - load_start_time
            print(f"模型加载完成，耗时: {load_time:.2f} 秒")
            
            # 执行推理
            print(f"开始生成图像，提示词: '{args.prompt}'")
            inference_start_time = time.time()
            
            # 使用 ComfyUI 的方式进行推理
            from comfy.sd import CLIP
            from comfy.sample import sample
            from comfy.samplers import KSampler
            
            # 编码提示词
            clip_encoder = CLIP(clip)
            positive_prompt = clip_encoder.encode([args.prompt])
            negative_prompt = clip_encoder.encode([args.negative_prompt])
            
            # 创建潜空间
            latent = torch.zeros([1, 4, args.height // 8, args.width // 8], device="cuda")
            
            # 设置采样器
            sampler = KSampler(model)
            samples = sampler.sample(
                positive_prompt, 
                negative_prompt, 
                latent, 
                args.steps, 
                7.0,  # cfg_scale
                "euler_ancestral",  # sampler_name
                "normal",  # scheduler
                1.0  # denoise
            )
            
            # 解码图像
            from comfy.utils import latent_to_image
            images = latent_to_image(vae, samples)
            
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
