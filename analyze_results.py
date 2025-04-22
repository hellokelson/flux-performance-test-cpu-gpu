#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(file_path):
    """加载测试结果文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载 {file_path} 时出错: {e}")
        return []

def analyze_results(cpu_results, gpu_results):
    """分析CPU和GPU测试结果"""
    # 创建结果字典
    analysis = {
        "cpu": {},
        "gpu": {},
        "comparison": {}
    }
    
    # 处理CPU结果
    for result in cpu_results:
        steps = result.get("steps")
        if steps:
            analysis["cpu"][steps] = {
                "total_time": result.get("total_time", 0),
                "generation_time": result.get("generation_time", 0),
                "time_per_step": result.get("time_per_step", 0),
                "load_time": result.get("load_time", 0)
            }
    
    # 处理GPU结果
    for result in gpu_results:
        steps = result.get("steps")
        if steps:
            analysis["gpu"][steps] = {
                "total_time": result.get("total_time", 0),
                "generation_time": result.get("generation_time", 0),
                "time_per_step": result.get("time_per_step", 0),
                "load_time": result.get("load_time", 0)
            }
    
    # 计算加速比
    for steps in analysis["cpu"]:
        if steps in analysis["gpu"]:
            cpu_time = analysis["cpu"][steps]["time_per_step"]
            gpu_time = analysis["gpu"][steps]["time_per_step"]
            
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
            else:
                speedup = 0
                
            analysis["comparison"][steps] = {
                "speedup": speedup,
                "cpu_time": cpu_time,
                "gpu_time": gpu_time
            }
    
    return analysis

def create_visualization(analysis):
    """创建可视化图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准备数据
    steps = sorted(analysis["comparison"].keys())
    cpu_times = [analysis["cpu"][s]["time_per_step"] for s in steps]
    gpu_times = [analysis["gpu"][s]["time_per_step"] for s in steps]
    speedups = [analysis["comparison"][s]["speedup"] for s in steps]
    
    # 绘制每步时间对比图
    x = np.arange(len(steps))
    width = 0.35
    
    ax1.bar(x - width/2, cpu_times, width, label='CPU (M7i.8xlarge + AMX)')
    ax1.bar(x + width/2, gpu_times, width, label='GPU (G6.4xlarge)')
    
    ax1.set_xlabel('步数')
    ax1.set_ylabel('每步时间 (秒)')
    ax1.set_title('CPU vs GPU 每步时间对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(steps)
    ax1.legend()
    
    # 为每个柱子添加数值标签
    for i, v in enumerate(cpu_times):
        ax1.text(i - width/2, v + 0.1, f'{v:.2f}s', ha='center')
    
    for i, v in enumerate(gpu_times):
        ax1.text(i + width/2, v + 0.1, f'{v:.2f}s', ha='center')
    
    # 绘制加速比图
    ax2.bar(steps, speedups, color='green')
    ax2.set_xlabel('步数')
    ax2.set_ylabel('加速比 (CPU时间/GPU时间)')
    ax2.set_title('GPU相对于CPU的加速比')
    
    # 为每个柱子添加数值标签
    for i, v in enumerate(speedups):
        ax2.text(steps[i], v + 0.1, f'{v:.2f}x', ha='center')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('flux_performance_comparison.png', dpi=300)
    print("性能对比图表已保存到 flux_performance_comparison.png")

def main():
    # 加载测试结果
    cpu_results = load_results('cpu_test_results.json')
    gpu_results = load_results('gpu_test_results.json')
    
    if not cpu_results or not gpu_results:
        print("错误: 无法加载测试结果")
        return
    
    # 分析结果
    analysis = analyze_results(cpu_results, gpu_results)
    
    # 创建可视化
    create_visualization(analysis)
    
    # 保存分析结果
    with open('performance_summary.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("性能分析完成，结果已保存到 performance_summary.json")
    
    # 打印摘要
    print("\n===== 性能测试摘要 =====")
    for steps in sorted(analysis["comparison"].keys()):
        speedup = analysis["comparison"][steps]["speedup"]
        cpu_time = analysis["comparison"][steps]["cpu_time"]
        gpu_time = analysis["comparison"][steps]["gpu_time"]
        
        print(f"步数: {steps}")
        print(f"  CPU每步时间: {cpu_time:.2f}秒")
        print(f"  GPU每步时间: {gpu_time:.2f}秒")
        print(f"  加速比: {speedup:.2f}x")
    
    print("=======================")

if __name__ == "__main__":
    main()
