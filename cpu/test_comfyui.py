#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUX.1-dev 模型 CPU 推理性能测试脚本
专门用于测试 black-forest-labs/FLUX.1-dev 模型
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
import torch
from PIL import Image
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1-dev CPU 推理测试")
    parser.add_argument("--prompt", type=str, default="一只可爱的小猫咪在草地上玩耍", help="生成图像的提示词")
    parser.add_argument("--negative_prompt", type=str, default="模糊的, 低质量的", help="负面提示词")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--precision", type=str, default="half", choices=["full", "half"], help="模型精度 (full=float32, half=float16)")
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
    
    # 设置 PyTorch 线程数以充分利用 CPU
    torch.set_num_threads(psutil.cpu_count(logical=True))
    print(f"PyTorch 线程数: {torch.get_num_threads()}")
    
    # 开始资源监控
    stop_monitor = threading.Event()
    resource_data = {'cpu_avg': 0, 'cpu_max': 0, 'memory_avg': 0, 'memory_max': 0}
    
    monitor_thread = threading.Thread(
        target=lambda: resource_data.update(monitor_resources(interval=0.5, stop_event=stop_monitor))
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # 使用 diffusers 库直接加载模型
        print(f"开始加载 FLUX.1-dev 模型，精度: {args.precision}")
        load_start_time = time.time()
        
        # 根据指定精度设置 dtype
        if args.precision == "full":
            dtype = torch.float32
        else:  # half
            dtype = torch.float16
        
        # 使用 diffusers 库加载模型
        from diffusers import DiffusionPipeline
        
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None
        )
        
        # 确保模型在 CPU 上
        pipe = pipe.to("cpu")
        
        load_time = time.time() - load_start_time
        print(f"模型加载完成，耗时: {load_time:.2f} 秒")
        
        # 执行推理
        print(f"开始生成图像，提示词: '{args.prompt}'")
        inference_start_time = time.time()
        
        # 使用 diffusers 进行推理
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=7.0
        ).images[0]
        
        inference_time = time.time() - inference_start_time
        print(f"图像生成完成，耗时: {inference_time:.2f} 秒")
        
        # 保存图像
        image_path = os.path.join(args.output_dir, f"flux_image_{args.precision}.png")
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
        output_file = os.path.join(args.output_dir, f"flux_performance_{args.precision}_metrics.json")
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

if __name__ == "__main__":
    main()
