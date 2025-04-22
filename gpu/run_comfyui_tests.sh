#!/bin/bash
set -e

echo "运行 FLUX.1-dev 模型 GPU 性能测试..."

# 进入 ComfyUI 目录
cd comfyui/ComfyUI

# 激活虚拟环境
source venv/bin/activate

# 返回到原始目录
cd ../../

# 创建输出目录
mkdir -p outputs

# 运行 float16 (half) 精度测试
echo "运行 float16 精度测试..."
python3 test_comfyui.py --precision half --output_dir ./outputs --steps 30 --height 768 --width 768

# 运行 float32 (full) 精度测试
echo "运行 float32 精度测试..."
python3 test_comfyui.py --precision full --output_dir ./outputs --steps 30 --height 768 --width 768

# 生成比较报告
echo "生成性能比较报告..."
python3 -c "
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 加载性能指标
half_metrics_path = './outputs/flux_performance_half_metrics.json'
full_metrics_path = './outputs/flux_performance_full_metrics.json'

if os.path.exists(half_metrics_path) and os.path.exists(full_metrics_path):
    with open(half_metrics_path, 'r') as f:
        half_metrics = json.load(f)
    
    with open(full_metrics_path, 'r') as f:
        full_metrics = json.load(f)
    
    # 创建比较图表
    labels = ['加载时间 (秒)', '推理时间 (秒)', 'GPU 使用率 (%)', 'GPU 内存使用率 (%)']
    half_values = [
        half_metrics.get('load_time', 0),
        half_metrics.get('inference_time', 0),
        half_metrics.get('gpu_avg', 0),
        half_metrics.get('gpu_memory_avg', 0)
    ]
    
    full_values = [
        full_metrics.get('load_time', 0),
        full_metrics.get('inference_time', 0),
        full_metrics.get('gpu_avg', 0),
        full_metrics.get('gpu_memory_avg', 0)
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
    rects1 = ax.bar(x - width/2, half_values, width, label='float16')
    rects2 = ax.bar(x + width/2, full_values, width, label='float32')
    
    ax.set_title('FLUX.1-dev 模型在 GPU 上的性能比较')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('./outputs/precision_comparison.png')
    print('已生成性能比较图表: ./outputs/precision_comparison.png')
    
    # 创建性能报告
    report = f'''# FLUX.1-dev 模型 GPU 性能测试报告

## 测试环境

- GPU: {half_metrics.get('gpu_name', 'Unknown')}
- 图像分辨率: {half_metrics.get('image_resolution', '768x768')}
- 推理步数: {half_metrics.get('steps', 30)}

## 性能比较

| 指标 | float16 (half) | float32 (full) | 差异比例 |
|------|---------------|---------------|---------|
| 模型加载时间 | {half_metrics.get('load_time', 0):.2f} 秒 | {full_metrics.get('load_time', 0):.2f} 秒 | {(full_metrics.get('load_time', 0) / half_metrics.get('load_time', 1) - 1) * 100:.2f}% |
| 推理时间 | {half_metrics.get('inference_time', 0):.2f} 秒 | {full_metrics.get('inference_time', 0):.2f} 秒 | {(full_metrics.get('inference_time', 0) / half_metrics.get('inference_time', 1) - 1) * 100:.2f}% |
| GPU 平均使用率 | {half_metrics.get('gpu_avg', 0):.2f}% | {full_metrics.get('gpu_avg', 0):.2f}% | {(full_metrics.get('gpu_avg', 0) / half_metrics.get('gpu_avg', 1) - 1) * 100:.2f}% |
| GPU 最大使用率 | {half_metrics.get('gpu_max', 0):.2f}% | {full_metrics.get('gpu_max', 0):.2f}% | {(full_metrics.get('gpu_max', 0) / half_metrics.get('gpu_max', 1) - 1) * 100:.2f}% |
| GPU 内存平均使用率 | {half_metrics.get('gpu_memory_avg', 0):.2f}% | {full_metrics.get('gpu_memory_avg', 0):.2f}% | {(full_metrics.get('gpu_memory_avg', 0) / half_metrics.get('gpu_memory_avg', 1) - 1) * 100:.2f}% |
| GPU 内存最大使用率 | {half_metrics.get('gpu_memory_max', 0):.2f}% | {full_metrics.get('gpu_memory_max', 0):.2f}% | {(full_metrics.get('gpu_memory_max', 0) / half_metrics.get('gpu_memory_max', 1) - 1) * 100:.2f}% |
| CPU 平均使用率 | {half_metrics.get('cpu_avg', 0):.2f}% | {full_metrics.get('cpu_avg', 0):.2f}% | {(full_metrics.get('cpu_avg', 0) / half_metrics.get('cpu_avg', 1) - 1) * 100:.2f}% |
| CPU 最大使用率 | {half_metrics.get('cpu_max', 0):.2f}% | {full_metrics.get('cpu_max', 0):.2f}% | {(full_metrics.get('cpu_max', 0) / half_metrics.get('cpu_max', 1) - 1) * 100:.2f}% |

## 结论

- float16 (half) 精度相比 float32 (full) 精度在加载时间上{'快' if half_metrics.get('load_time', 0) < full_metrics.get('load_time', 0) else '慢'} {abs(half_metrics.get('load_time', 0) / full_metrics.get('load_time', 1) - 1) * 100:.2f}%
- float16 (half) 精度相比 float32 (full) 精度在推理时间上{'快' if half_metrics.get('inference_time', 0) < full_metrics.get('inference_time', 0) else '慢'} {abs(half_metrics.get('inference_time', 0) / full_metrics.get('inference_time', 1) - 1) * 100:.2f}%
- float16 (half) 精度相比 float32 (full) 精度在 GPU 内存使用上{'少' if half_metrics.get('gpu_memory_avg', 0) < full_metrics.get('gpu_memory_avg', 0) else '多'} {abs(half_metrics.get('gpu_memory_avg', 0) / full_metrics.get('gpu_memory_avg', 1) - 1) * 100:.2f}%

## 建议

- {'对于 FLUX.1-dev 模型，推荐使用 float16 (half) 精度，可以获得更好的性能表现。' if half_metrics.get('inference_time', 0) < full_metrics.get('inference_time', 0) else '对于 FLUX.1-dev 模型，如果追求更高的生成质量，可以考虑使用 float32 (full) 精度。'}
- 在 GPU 资源有限的环境中，{'建议使用 float16 (half) 精度以节省 GPU 内存。' if half_metrics.get('gpu_memory_avg', 0) < full_metrics.get('gpu_memory_avg', 0) else '可以考虑使用更小的图像分辨率以节省 GPU 内存。'}

## 图像质量比较

请查看 ./outputs 目录下的 flux_image_half.png 和 flux_image_full.png 文件，比较不同精度下生成图像的质量差异。
'''
    
    with open('./outputs/gpu_performance_report.md', 'w') as f:
        f.write(report)
    
    print('已生成性能测试报告: ./outputs/gpu_performance_report.md')
else:
    print('未找到性能指标文件，无法生成比较报告')
"

echo "测试完成！结果保存在 ./outputs 目录中"
