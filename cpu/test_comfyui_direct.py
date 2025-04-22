#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUX.1-dev 模型 CPU 推理性能测试脚本 (直接使用 ComfyUI Python API)
支持 Intel AMX 加速器测试
"""

import os
import time
import argparse
import psutil
import json
import subprocess
import threading
import numpy as np
import sys
import importlib.util

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1-dev ComfyUI CPU 推理测试 (直接API)")
    parser.add_argument("--prompt", type=str, default="一只可爱的小猫咪在草地上玩耍", help="生成图像的提示词")
    parser.add_argument("--negative_prompt", type=str, default="模糊的, 低质量的", help="负面提示词")
    parser.add_argument("--steps", type=int, default=30, help="推理步数")
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--precision", type=str, default="half", choices=["full", "half"], help="模型精度 (full=float32, half=float16)")
    parser.add_argument("--comfyui_dir", type=str, default="./comfyui/ComfyUI", help="ComfyUI 目录")
    return parser.parse_args()

def check_amx_support():
    """检查系统是否支持 Intel AMX"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'amx_bf16' in cpuinfo or 'amx_tile' in cpuinfo or 'amx_int8' in cpuinfo or 'amx' in cpuinfo:
                print("系统支持 Intel AMX 加速器")
                return True
        print("系统不支持 Intel AMX 加速器")
        return False
    except Exception as e:
        print(f"检查 AMX 支持时出错: {e}")
        return False

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
    
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        cpu_percentages.append(cpu_percent)
        memory_usages.append(memory.percent)
        
        time.sleep(interval)
            
    return {
        'cpu_avg': np.mean(cpu_percentages) if cpu_percentages else 0,
        'cpu_max': np.max(cpu_percentages) if cpu_percentages else 0,
        'memory_avg': np.mean(memory_usages) if memory_usages else 0,
        'memory_max': np.max(memory_usages) if memory_usages else 0
    }

def save_metrics(metrics, output_file):
    """保存性能指标"""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查 Intel AMX 支持
    has_amx = check_amx_support()
    
    # 记录 CPU 信息
    cpu_info = get_cpu_info()
    print(f"CPU 信息: {cpu_info['brand_raw']}, {cpu_info['count']} 核心")
    
    # 添加 ComfyUI 路径
    comfyui_path = os.path.abspath(args.comfyui_dir)
    sys.path.append(comfyui_path)
    
    # 导入 ComfyUI 模块
    try:
        # 检查模型文件
        model_dir = os.path.join(comfyui_path, "models/checkpoints")
        model_path = os.path.join(model_dir, "flux_1_dev.safetensors")
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            # 查找其他模型
            models = [f for f in os.listdir(model_dir) if f.endswith('.safetensors') or f.endswith('.ckpt')]
            if models:
                print(f"找到其他模型: {models}")
                model_path = os.path.join(model_dir, models[0])
                print(f"使用模型: {model_path}")
            else:
                print("未找到任何模型文件")
                return
        
        # 开始资源监控
        stop_monitor = threading.Event()
        resource_data = {'cpu_avg': 0, 'cpu_max': 0, 'memory_avg': 0, 'memory_max': 0}
        
        monitor_thread = threading.Thread(
            target=lambda: resource_data.update(monitor_resources(interval=0.5, stop_event=stop_monitor))
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # 导入必要的模块
            import torch
            import folder_paths
            
            # 设置模型路径
            folder_paths.add_model_folder_path("checkpoints", model_dir)
            
            # 设置精度
            if args.precision == "full":
                dtype = torch.float32
            else:
                dtype = torch.float16
            
            # 加载模型
            print(f"加载模型: {model_path}")
            load_start_time = time.time()
            
            # 导入模型加载函数
            from comfy.sd import load_checkpoint_guess_config
            model, clip, vae = load_checkpoint_guess_config(model_path, output_vae=True, output_clip=True, embedding_directory=None, dtype=dtype)
            
            load_time = time.time() - load_start_time
            print(f"模型加载完成，耗时: {load_time:.2f} 秒")
            
            # 生成图像
            print(f"开始生成图像，提示词: '{args.prompt}'")
            inference_start_time = time.time()
            
            # 编码提示词
            from comfy.sd import CLIP
            clip_encoder = CLIP(clip)
            positive_prompt = clip_encoder.encode([args.prompt])
            negative_prompt = clip_encoder.encode([args.negative_prompt])
            
            # 创建潜空间
            from comfy.sample import prepare_sampling
            from comfy.samplers import KSampler
            
            latent = torch.zeros([1, 4, args.height // 8, args.width // 8])
            
            # 设置采样器
            sampler = KSampler(model)
            samples = sampler.sample(positive_prompt, negative_prompt, latent, args.steps, 7.0, "euler_ancestral", "normal", 1.0)
            
            # 解码图像
            from comfy.utils import latent_to_image
            images = latent_to_image(vae, samples)
            
            inference_time = time.time() - inference_start_time
            print(f"图像生成完成，耗时: {inference_time:.2f} 秒")
            
            # 保存图像
            from PIL import Image
            image = Image.fromarray(images[0])
            image_path = os.path.join(args.output_dir, f"image_{args.precision}.png")
            image.save(image_path)
            print(f"图像已保存到: {image_path}")
            
            # 停止资源监控
            stop_monitor.set()
            monitor_thread.join(timeout=1)
            
            # 记录性能指标
            metrics = {
                'device': 'CPU',
                'model': 'FLUX.1-dev',
                'precision': args.precision,
                'has_amx': has_amx,
                'load_time': load_time,
                'inference_time': inference_time,
                'cpu_avg': resource_data['cpu_avg'],
                'cpu_max': resource_data['cpu_max'],
                'memory_avg': resource_data['memory_avg'],
                'memory_max': resource_data['memory_max'],
                'image_resolution': f"{args.width}x{args.height}",
                'steps': args.steps
            }
            
            # 保存性能指标
            output_file = os.path.join(args.output_dir, f"comfyui_direct_{args.precision}_metrics.json")
            save_metrics(metrics, output_file)
            
            print("\n性能测试结果:")
            print(f"模型加载时间: {load_time:.2f} 秒")
            print(f"推理时间: {inference_time:.2f} 秒")
            print(f"CPU 平均使用率: {resource_data['cpu_avg']:.2f}%")
            print(f"CPU 最大使用率: {resource_data['cpu_max']:.2f}%")
            print(f"内存平均使用率: {resource_data['memory_avg']:.2f}%")
            print(f"内存最大使用率: {resource_data['memory_max']:.2f}%")
            
            print(f"\n性能指标已保存到 {output_file}")
            
        except Exception as e:
            print(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保停止资源监控
            stop_monitor.set()
    
    except ImportError as e:
        print(f"导入 ComfyUI 模块失败: {e}")
        print("请确保 ComfyUI 已正确安装")

if __name__ == "__main__":
    main()