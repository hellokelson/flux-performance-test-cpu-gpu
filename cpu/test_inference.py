#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUX.1-dev 模型 CPU 推理性能测试脚本
"""

import os
import time
import argparse
import psutil
import torch
import numpy as np
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
import sys

# 添加通用模块路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from utils import log_metrics, save_image, get_cpu_info

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1-dev CPU 推理测试")
    parser.add_argument("--prompt", type=str, default="一只可爱的小猫咪在草地上玩耍", help="生成图像的提示词")
    parser.add_argument("--negative_prompt", type=str, default="模糊的, 低质量的", help="负面提示词")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="推理步数")
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--model_path", type=str, default="./models/FLUX.1-dev", help="模型路径")
    return parser.parse_args()

def monitor_resources(interval=1.0, duration=None):
    """监控系统资源使用情况"""
    cpu_percentages = []
    memory_usages = []
    
    start_time = time.time()
    while True:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        cpu_percentages.append(cpu_percent)
        memory_usages.append(memory.percent)
        
        time.sleep(interval)
        
        if duration and time.time() - start_time >= duration:
            break
            
    return {
        'cpu_avg': np.mean(cpu_percentages),
        'cpu_max': np.max(cpu_percentages),
        'memory_avg': np.mean(memory_usages),
        'memory_max': np.max(memory_usages)
    }

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录 CPU 信息
    cpu_info = get_cpu_info()
    print(f"CPU 信息: {cpu_info['brand_raw']}, {cpu_info['count']} 核心")
    
    # 记录开始加载模型的时间
    load_start_time = time.time()
    
    print(f"正在从 {args.model_path} 加载 FLUX.1-dev 模型...")
    
    # 加载模型
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        use_safetensors=True,
        torch_dtype=torch.float32  # CPU 使用 float32
    )
    
    # 确保模型在 CPU 上
    pipe = pipe.to("cpu")
    
    # 记录模型加载时间
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    print(f"模型加载完成，耗时: {load_time:.2f} 秒")
    
    # 开始资源监控
    import threading
    resource_data = {'cpu_avg': 0, 'cpu_max': 0, 'memory_avg': 0, 'memory_max': 0}
    
    def resource_monitor_thread():
        nonlocal resource_data
        resource_data = monitor_resources(interval=0.5, duration=None)
    
    monitor_thread = threading.Thread(target=resource_monitor_thread)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 执行推理
    print(f"开始生成图像，提示词: '{args.prompt}'")
    inference_times = []
    
    for i in range(args.batch_size):
        inference_start = time.time()
        
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width
        ).images[0]
        
        inference_end = time.time()
        inference_time = inference_end - inference_start
        inference_times.append(inference_time)
        
        print(f"图像 {i+1}/{args.batch_size} 生成完成，耗时: {inference_time:.2f} 秒")
        
        # 保存图像
        image_path = os.path.join(args.output_dir, f"image_{i+1}.png")
        save_image(image, image_path)
    
    # 停止资源监控
    monitor_thread.join(timeout=1)
    
    # 计算并打印性能指标
    avg_inference_time = np.mean(inference_times)
    
    print("\n性能测试结果:")
    print(f"模型加载时间: {load_time:.2f} 秒")
    print(f"平均推理时间: {avg_inference_time:.2f} 秒/图像")
    print(f"CPU 平均使用率: {resource_data['cpu_avg']:.2f}%")
    print(f"CPU 最大使用率: {resource_data['cpu_max']:.2f}%")
    print(f"内存平均使用率: {resource_data['memory_avg']:.2f}%")
    print(f"内存最大使用率: {resource_data['memory_max']:.2f}%")
    
    # 记录性能指标
    metrics = {
        'device': 'CPU',
        'model': 'FLUX.1-dev',
        'load_time': load_time,
        'avg_inference_time': avg_inference_time,
        'cpu_avg': resource_data['cpu_avg'],
        'cpu_max': resource_data['cpu_max'],
        'memory_avg': resource_data['memory_avg'],
        'memory_max': resource_data['memory_max'],
        'batch_size': args.batch_size,
        'image_resolution': f"{args.width}x{args.height}",
        'inference_steps': args.num_inference_steps
    }
    
    # 保存性能指标
    log_metrics(metrics, os.path.join(args.output_dir, "cpu_performance_metrics.json"))
    
    print(f"\n所有图像已保存到 {args.output_dir} 目录")

if __name__ == "__main__":
    main()
