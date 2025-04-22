#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(file_path):
    """加载测试结果"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_results(cpu_results, gpu_results):
    """分析测试结果"""
    summary = {
        "cpu": {},
        "gpu": {},
        "speedup": {}
    }
    
    # 处理CPU结果
    for result in cpu_results:
        steps = result["steps"]
        dtype = result.get("dtype", "float32")  # 默认为float32
        key = f"{steps}steps_{dtype}"
        
        summary["cpu"][key] = {
            "total_time": result["total_time"],
            "generation_time": result["generation_time"],
            "time_per_step": result["time_per_step"],
            "load_time": result["load_time"]
        }
    
    # 处理GPU结果
    for result in gpu_results:
        steps = result["steps"]
        dtype = result.get("dtype", "float16")  # 默认为float16
        key = f"{steps}steps_{dtype}"
        
        summary["gpu"][key] = {
            "total_time": result["total_time"],
            "generation_time": result["generation_time"],
            "time_per_step": result["time_per_step"],
            "load_time": result["load_time"]
        }
    
    # 计算加速比
    for cpu_key, cpu_data in summary["cpu"].items():
        steps = cpu_key.split("steps_")[0] + "steps"
        cpu_dtype = cpu_key.split("_")[1]
        
        # 对于CPU的float32，与GPU的float16和float32比较
        if cpu_dtype == "float32":
            gpu_key_f16 = f"{steps}_float16"
            gpu_key_f32 = f"{steps}_float32"
            
            if gpu_key_f16 in summary["gpu"]:
                speedup_f16 = cpu_data["generation_time"] / summary["gpu"][gpu_key_f16]["generation_time"]
                summary["speedup"][f"{steps}_cpu_f32_vs_gpu_f16"] = speedup_f16
            
            if gpu_key_f32 in summary["gpu"]:
                speedup_f32 = cpu_data["generation_time"] / summary["gpu"][gpu_key_f32]["generation_time"]
                summary["speedup"][f"{steps}_cpu_f32_vs_gpu_f32"] = speedup_f32
        
        # 对于CPU的bfloat16，与GPU的float16和float32比较
        elif cpu_dtype == "bfloat16":
            gpu_key_f16 = f"{steps}_float16"
            gpu_key_f32 = f"{steps}_float32"
            
            if gpu_key_f16 in summary["gpu"]:
                speedup_f16 = cpu_data["generation_time"] / summary["gpu"][gpu_key_f16]["generation_time"]
                summary["speedup"][f"{steps}_cpu_bf16_vs_gpu_f16"] = speedup_f16
            
            if gpu_key_f32 in summary["gpu"]:
                speedup_f32 = cpu_data["generation_time"] / summary["gpu"][gpu_key_f32]["generation_time"]
                summary["speedup"][f"{steps}_cpu_bf16_vs_gpu_f32"] = speedup_f32
            
            # 与CPU的float32比较
            cpu_key_f32 = f"{steps}_float32"
            if cpu_key_f32 in summary["cpu"]:
                speedup_cpu = summary["cpu"][cpu_key_f32]["generation_time"] / cpu_data["generation_time"]
                summary["speedup"][f"{steps}_cpu_f32_vs_cpu_bf16"] = speedup_cpu
    
    return summary

def create_visualization(summary):
    """创建可视化图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 准备数据
    steps_list = [5, 20]
    cpu_f32_times = []
    cpu_bf16_times = []
    gpu_f16_times = []
    gpu_f32_times = []
    
    for steps in steps_list:
        # CPU float32
        key = f"{steps}steps_float32"
        if key in summary["cpu"]:
            cpu_f32_times.append(summary["cpu"][key]["time_per_step"])
        else:
            cpu_f32_times.append(0)
        
        # CPU bfloat16
        key = f"{steps}steps_bfloat16"
        if key in summary["cpu"]:
            cpu_bf16_times.append(summary["cpu"][key]["time_per_step"])
        else:
            cpu_bf16_times.append(0)
        
        # GPU float16
        key = f"{steps}steps_float16"
        if key in summary["gpu"]:
            gpu_f16_times.append(summary["gpu"][key]["time_per_step"])
        else:
            gpu_f16_times.append(0)
        
        # GPU float32
        key = f"{steps}steps_float32"
        if key in summary["gpu"]:
            gpu_f32_times.append(summary["gpu"][key]["time_per_step"])
        else:
            gpu_f32_times.append(0)
    
    # 绘制每步时间柱状图
    x = np.arange(len(steps_list))
    width = 0.2
    
    ax1.bar(x - width*1.5, cpu_f32_times, width, label='CPU (float32 AMX)')
    if any(cpu_bf16_times):  # 只有在有bfloat16数据时才绘制
        ax1.bar(x - width/2, cpu_bf16_times, width, label='CPU (bfloat16 AMX)')
    ax1.bar(x + width/2, gpu_f16_times, width, label='GPU (float16)')
    ax1.bar(x + width*1.5, gpu_f32_times, width, label='GPU (float32)')
    
    ax1.set_xlabel('步数')
    ax1.set_ylabel('每步时间 (秒)')
    ax1.set_title('FLUX.1-dev 模型每步推理时间')
    ax1.set_xticks(x)
    ax1.set_xticklabels(steps_list)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制加速比柱状图
    speedup_labels = []
    speedup_values = []
    
    for key, value in summary["speedup"].items():
        speedup_labels.append(key)
        speedup_values.append(value)
    
    ax2.bar(range(len(speedup_values)), speedup_values, color='skyblue')
    ax2.set_xlabel('比较项')
    ax2.set_ylabel('加速比')
    ax2.set_title('FLUX.1-dev 模型性能加速比')
    ax2.set_xticks(range(len(speedup_labels)))
    ax2.set_xticklabels(speedup_labels, rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(speedup_values):
        ax2.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('flux_performance_comparison.png', dpi=300)
    print("已保存性能对比图表到: flux_performance_comparison.png")
    
    # 保存性能总结
    with open('performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("已保存性能总结到: performance_summary.json")

def main():
    # 加载测试结果
    cpu_results = load_results('cpu_test_results.json')
    gpu_results = load_results('gpu_test_results.json')
    
    if not cpu_results and not gpu_results:
        print("错误: 找不到任何测试结果文件")
        return
    
    # 分析结果
    summary = analyze_results(cpu_results, gpu_results)
    
    # 创建可视化
    create_visualization(summary)
    
    # 打印总结
    print("\n===== FLUX.1-dev 模型性能测试总结 =====")
    
    # 打印CPU结果
    print("\nCPU 测试结果:")
    for key, data in summary["cpu"].items():
        print(f"  {key}: 每步时间 = {data['time_per_step']:.2f}秒, 总生成时间 = {data['generation_time']:.2f}秒")
    
    # 打印GPU结果
    print("\nGPU 测试结果:")
    for key, data in summary["gpu"].items():
        print(f"  {key}: 每步时间 = {data['time_per_step']:.2f}秒, 总生成时间 = {data['generation_time']:.2f}秒")
    
    # 打印加速比
    print("\n加速比:")
    for key, value in summary["speedup"].items():
        print(f"  {key}: {value:.2f}x")

if __name__ == "__main__":
    main()
