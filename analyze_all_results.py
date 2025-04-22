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

def analyze_results(cpu_results, gpu_results, arm_cpu_results):
    """分析测试结果"""
    summary = {
        "x86_cpu": {},
        "x86_gpu": {},
        "arm_cpu": {},
        "speedup": {}
    }
    
    # 处理x86 CPU结果
    for result in cpu_results:
        steps = result["steps"]
        dtype = result.get("dtype", "bfloat16")  # 默认为bfloat16
        batch_size = result.get("batch_size", 4)
        resolution = result.get("resolution", "1024x1024")
        key = f"{steps}steps_{dtype}_{batch_size}batch_{resolution}"
        
        summary["x86_cpu"][key] = {
            "total_time": result["total_time"],
            "generation_time": result["generation_time"],
            "time_per_step": result["time_per_step"],
            "time_per_image": result.get("time_per_image", result["generation_time"] / batch_size),
            "load_time": result["load_time"]
        }
    
    # 处理x86 GPU结果
    for result in gpu_results:
        steps = result["steps"]
        dtype = result.get("dtype", "float16")  # 默认为float16
        batch_size = result.get("batch_size", 4)
        resolution = result.get("resolution", "1024x1024")
        key = f"{steps}steps_{dtype}_{batch_size}batch_{resolution}"
        
        summary["x86_gpu"][key] = {
            "total_time": result["total_time"],
            "generation_time": result["generation_time"],
            "time_per_step": result["time_per_step"],
            "time_per_image": result.get("time_per_image", result["generation_time"] / batch_size),
            "load_time": result["load_time"]
        }
    
    # 处理ARM CPU结果
    for result in arm_cpu_results:
        steps = result["steps"]
        dtype = result.get("dtype", "float32")  # 默认为float32
        batch_size = result.get("batch_size", 4)
        resolution = result.get("resolution", "1024x1024")
        key = f"{steps}steps_{dtype}_{batch_size}batch_{resolution}"
        
        summary["arm_cpu"][key] = {
            "total_time": result["total_time"],
            "generation_time": result["generation_time"],
            "time_per_step": result["time_per_step"],
            "time_per_image": result.get("time_per_image", result["generation_time"] / batch_size),
            "load_time": result["load_time"]
        }
    
    # 计算加速比
    # x86 CPU vs x86 GPU
    for cpu_key, cpu_data in summary["x86_cpu"].items():
        parts = cpu_key.split("_")
        steps = parts[0]
        cpu_dtype = parts[1]
        batch_size = parts[2]
        resolution = parts[3]
        
        # x86 CPU (bfloat16) vs x86 GPU (float16)
        gpu_key = f"{steps}steps_float16_{batch_size}_{resolution}"
        if gpu_key in summary["x86_gpu"]:
            speedup = cpu_data["time_per_image"] / summary["x86_gpu"][gpu_key]["time_per_image"]
            summary["speedup"][f"{steps}_x86_cpu_bf16_vs_x86_gpu_f16"] = speedup
    
    # x86 CPU vs ARM CPU
    for x86_key, x86_data in summary["x86_cpu"].items():
        parts = x86_key.split("_")
        steps = parts[0]
        x86_dtype = parts[1]
        batch_size = parts[2]
        resolution = parts[3]
        
        # x86 CPU (bfloat16) vs ARM CPU (float32)
        arm_key = f"{steps}steps_float32_{batch_size}_{resolution}"
        if arm_key in summary["arm_cpu"]:
            speedup = summary["arm_cpu"][arm_key]["time_per_image"] / x86_data["time_per_image"]
            summary["speedup"][f"{steps}_arm_cpu_f32_vs_x86_cpu_bf16"] = speedup
    
    # ARM CPU vs x86 GPU
    for arm_key, arm_data in summary["arm_cpu"].items():
        parts = arm_key.split("_")
        steps = parts[0]
        arm_dtype = parts[1]
        batch_size = parts[2]
        resolution = parts[3]
        
        # ARM CPU (float32) vs x86 GPU (float16)
        gpu_key = f"{steps}steps_float16_{batch_size}_{resolution}"
        if gpu_key in summary["x86_gpu"]:
            speedup = arm_data["time_per_image"] / summary["x86_gpu"][gpu_key]["time_per_image"]
            summary["speedup"][f"{steps}_arm_cpu_f32_vs_x86_gpu_f16"] = speedup
    
    return summary

def create_visualization(summary):
    """创建可视化图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 准备数据
    steps_list = [5, 20]
    x86_cpu_bf16_times = []
    arm_cpu_f32_times = []
    x86_gpu_f16_times = []
    
    for steps in steps_list:
        # x86 CPU bfloat16
        key = f"{steps}steps_bfloat16_4batch_1024x1024"
        if key in summary["x86_cpu"]:
            x86_cpu_bf16_times.append(summary["x86_cpu"][key]["time_per_image"])
        else:
            x86_cpu_bf16_times.append(0)
        
        # ARM CPU float32
        key = f"{steps}steps_float32_4batch_1024x1024"
        if key in summary["arm_cpu"]:
            arm_cpu_f32_times.append(summary["arm_cpu"][key]["time_per_image"])
        else:
            arm_cpu_f32_times.append(0)
        
        # x86 GPU float16
        key = f"{steps}steps_float16_4batch_1024x1024"
        if key in summary["x86_gpu"]:
            x86_gpu_f16_times.append(summary["x86_gpu"][key]["time_per_image"])
        else:
            x86_gpu_f16_times.append(0)
    
    # 绘制每张图片时间柱状图
    x = np.arange(len(steps_list))
    width = 0.25
    
    ax1.bar(x - width, x86_cpu_bf16_times, width, label='x86 CPU (bfloat16 AMX)')
    ax1.bar(x, arm_cpu_f32_times, width, label='ARM CPU (float32)')
    ax1.bar(x + width, x86_gpu_f16_times, width, label='x86 GPU (float16)')
    
    ax1.set_xlabel('步数')
    ax1.set_ylabel('每张图片时间 (秒)')
    ax1.set_title('FLUX.1-dev 模型每张图片生成时间 (1024x1024, batch=4)')
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
    plt.savefig('flux_all_performance_comparison.png', dpi=300)
    print("已保存性能对比图表到: flux_all_performance_comparison.png")
    
    # 保存性能总结
    with open('all_performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("已保存性能总结到: all_performance_summary.json")

def main():
    # 加载测试结果
    cpu_results = load_results('cpu_test_results.json')
    gpu_results = load_results('gpu_test_results.json')
    arm_cpu_results = load_results('arm_cpu_test_results.json')
    
    if not cpu_results and not gpu_results and not arm_cpu_results:
        print("错误: 找不到任何测试结果文件")
        return
    
    # 分析结果
    summary = analyze_results(cpu_results, gpu_results, arm_cpu_results)
    
    # 创建可视化
    create_visualization(summary)
    
    # 打印总结
    print("\n===== FLUX.1-dev 模型性能测试总结 (1024x1024, batch=4) =====")
    
    # 打印x86 CPU结果
    print("\nx86 CPU 测试结果:")
    for key, data in summary["x86_cpu"].items():
        print(f"  {key}: 每张图片时间 = {data['time_per_image']:.2f}秒, 总生成时间 = {data['generation_time']:.2f}秒")
    
    # 打印x86 GPU结果
    print("\nx86 GPU 测试结果:")
    for key, data in summary["x86_gpu"].items():
        print(f"  {key}: 每张图片时间 = {data['time_per_image']:.2f}秒, 总生成时间 = {data['generation_time']:.2f}秒")
    
    # 打印ARM CPU结果
    print("\nARM CPU 测试结果:")
    for key, data in summary["arm_cpu"].items():
        print(f"  {key}: 每张图片时间 = {data['time_per_image']:.2f}秒, 总生成时间 = {data['generation_time']:.2f}秒")
    
    # 打印加速比
    print("\n加速比:")
    for key, value in summary["speedup"].items():
        print(f"  {key}: {value:.2f}x")

if __name__ == "__main__":
    main()
