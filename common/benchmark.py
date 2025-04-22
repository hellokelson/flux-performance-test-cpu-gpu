#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能基准测试脚本
"""

import os
import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import log_metrics, format_size

def plot_performance_comparison(cpu_metrics_file, gpu_metrics_file, output_dir):
    """绘制 CPU 和 GPU 性能对比图"""
    # 检查文件是否存在
    if not os.path.exists(cpu_metrics_file) or not os.path.exists(gpu_metrics_file):
        print("错误: 性能指标文件不存在")
        return
    
    # 读取性能指标
    with open(cpu_metrics_file, 'r') as f:
        cpu_data = json.load(f)
        if not isinstance(cpu_data, list):
            cpu_data = [cpu_data]
    
    with open(gpu_metrics_file, 'r') as f:
        gpu_data = json.load(f)
        if not isinstance(gpu_data, list):
            gpu_data = [gpu_data]
    
    # 计算平均值
    cpu_load_time = np.mean([item.get('load_time', 0) for item in cpu_data])
    cpu_inference_time = np.mean([item.get('avg_inference_time', 0) for item in cpu_data])
    
    gpu_load_time = np.mean([item.get('load_time', 0) for item in gpu_data])
    gpu_inference_time = np.mean([item.get('avg_inference_time', 0) for item in gpu_data])
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 模型加载时间对比
    devices = ['CPU (M7i)', 'GPU (G6)']
    load_times = [cpu_load_time, gpu_load_time]
    
    ax1.bar(devices, load_times, color=['blue', 'green'])
    ax1.set_title('模型加载时间对比')
    ax1.set_ylabel('时间 (秒)')
    for i, v in enumerate(load_times):
        ax1.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # 推理时间对比
    inference_times = [cpu_inference_time, gpu_inference_time]
    
    ax2.bar(devices, inference_times, color=['blue', 'green'])
    ax2.set_title('平均推理时间对比')
    ax2.set_ylabel('时间 (秒)')
    for i, v in enumerate(inference_times):
        ax2.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # 保存图表
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_file)
    print(f"性能对比图已保存到 {output_file}")
    
    # 计算加速比
    if cpu_inference_time > 0 and gpu_inference_time > 0:
        speedup = cpu_inference_time / gpu_inference_time
        print(f"GPU 相对于 CPU 的加速比: {speedup:.2f}x")
    
    return {
        'cpu_load_time': cpu_load_time,
        'gpu_load_time': gpu_load_time,
        'cpu_inference_time': cpu_inference_time,
        'gpu_inference_time': gpu_inference_time,
        'speedup': speedup if 'speedup' in locals() else None
    }

def generate_report(cpu_metrics_file, gpu_metrics_file, output_dir):
    """生成性能测试报告"""
    comparison = plot_performance_comparison(cpu_metrics_file, gpu_metrics_file, output_dir)
    
    # 读取详细指标
    with open(cpu_metrics_file, 'r') as f:
        cpu_data = json.load(f)
        if isinstance(cpu_data, list):
            cpu_data = cpu_data[-1]  # 使用最新的数据
    
    with open(gpu_metrics_file, 'r') as f:
        gpu_data = json.load(f)
        if isinstance(gpu_data, list):
            gpu_data = gpu_data[-1]  # 使用最新的数据
    
    # 生成报告
    report = f"""# FLUX.1-dev 模型性能测试报告

## 测试环境

- CPU: M7i EC2 实例
- GPU: G6 EC2 实例
- 模型: black-forest-labs/FLUX.1-dev
- 图像分辨率: {cpu_data.get('image_resolution', 'N/A')}
- 推理步数: {cpu_data.get('inference_steps', 'N/A')}

## 性能对比

| 指标 | CPU (M7i) | GPU (G6) | 加速比 |
|------|-----------|----------|--------|
| 模型加载时间 | {comparison['cpu_load_time']:.2f}s | {comparison['gpu_load_time']:.2f}s | {comparison['cpu_load_time']/comparison['gpu_load_time']:.2f}x |
| 平均推理时间 | {comparison['cpu_inference_time']:.2f}s | {comparison['gpu_inference_time']:.2f}s | {comparison['speedup']:.2f}x |
| CPU 平均使用率 | {cpu_data.get('cpu_avg', 'N/A')}% | {gpu_data.get('cpu_avg', 'N/A')}% | N/A |
| 内存平均使用率 | {cpu_data.get('memory_avg', 'N/A')}% | {gpu_data.get('memory_avg', 'N/A')}% | N/A |

## GPU 特定指标

| 指标 | 值 |
|------|-----|
| GPU 型号 | {gpu_data.get('gpu_name', 'N/A')} |
| GPU 平均使用率 | {gpu_data.get('gpu_avg', 'N/A')}% |
| GPU 最大使用率 | {gpu_data.get('gpu_max', 'N/A')}% |
| GPU 内存平均使用率 | {gpu_data.get('gpu_memory_avg', 'N/A')}% |
| GPU 内存最大使用率 | {gpu_data.get('gpu_memory_max', 'N/A')}% |

## 结论

GPU (G6) 相比 CPU (M7i) 在推理速度上提升了 {comparison['speedup']:.2f} 倍。

"""
    
    # 保存报告
    report_file = os.path.join(output_dir, 'performance_report.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"性能测试报告已保存到 {report_file}")

def main():
    parser = argparse.ArgumentParser(description="FLUX.1-dev 性能对比分析")
    parser.add_argument("--cpu_metrics", type=str, default="../cpu/outputs/cpu_performance_metrics.json", help="CPU 性能指标文件路径")
    parser.add_argument("--gpu_metrics", type=str, default="../gpu/outputs/gpu_performance_metrics.json", help="GPU 性能指标文件路径")
    parser.add_argument("--output_dir", type=str, default="./reports", help="输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成报告
    generate_report(args.cpu_metrics, args.gpu_metrics, args.output_dir)

if __name__ == "__main__":
    main()
