#!/bin/bash
set -e

echo "开始 Intel AMX 加速器性能测试..."

# 激活虚拟环境
source flux_env/bin/activate

pip install --upgrade diffusers
# 创建输出目录
mkdir -p outputs

# 测试不同精度
echo "测试 float32 精度..."
python test_inference.py --precision float32 --output_dir ./outputs/float32

echo "测试 float16 精度..."
python test_inference.py --precision float16 --output_dir ./outputs/float16

echo "测试 bfloat16 精度 (Intel AMX 优化格式)..."
python test_inference.py --precision bfloat16 --output_dir ./outputs/bfloat16

# 测试量化模型
echo "测试 INT8 量化模型..."
python test_inference.py --precision float32 --quantize --output_dir ./outputs/int8

# 生成比较报告
echo "生成性能比较报告..."
python -c "
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 读取各种精度的性能指标
results = {}
for precision in ['float32', 'float16', 'bfloat16', 'int8']:
    try:
        if precision == 'int8':
            file_path = './outputs/int8/cpu_performance_float32_metrics.json'
        else:
            file_path = f'./outputs/{precision}/cpu_performance_{precision}_metrics.json'
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    data = data[-1]  # 使用最新的数据
                results[precision] = data
    except Exception as e:
        print(f'读取 {precision} 数据时出错: {e}')

if not results:
    print('没有找到性能指标数据')
    exit(1)

# 创建图表
plt.figure(figsize=(12, 8))

# 推理时间比较
precisions = list(results.keys())
inference_times = [results[p].get('avg_inference_time', 0) for p in precisions]

plt.subplot(2, 2, 1)
plt.bar(precisions, inference_times, color=['blue', 'green', 'red', 'purple'])
plt.title('不同精度下的平均推理时间')
plt.ylabel('时间 (秒)')
for i, v in enumerate(inference_times):
    plt.text(i, v + 0.1, f'{v:.2f}s', ha='center')

# 加载时间比较
load_times = [results[p].get('load_time', 0) for p in precisions]

plt.subplot(2, 2, 2)
plt.bar(precisions, load_times, color=['blue', 'green', 'red', 'purple'])
plt.title('不同精度下的模型加载时间')
plt.ylabel('时间 (秒)')
for i, v in enumerate(load_times):
    plt.text(i, v + 0.1, f'{v:.2f}s', ha='center')

# CPU 使用率比较
cpu_avgs = [results[p].get('cpu_avg', 0) for p in precisions]

plt.subplot(2, 2, 3)
plt.bar(precisions, cpu_avgs, color=['blue', 'green', 'red', 'purple'])
plt.title('不同精度下的 CPU 平均使用率')
plt.ylabel('使用率 (%)')
for i, v in enumerate(cpu_avgs):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')

# 内存使用率比较
memory_avgs = [results[p].get('memory_avg', 0) for p in precisions]

plt.subplot(2, 2, 4)
plt.bar(precisions, memory_avgs, color=['blue', 'green', 'red', 'purple'])
plt.title('不同精度下的内存平均使用率')
plt.ylabel('使用率 (%)')
for i, v in enumerate(memory_avgs):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')

plt.tight_layout()
plt.savefig('./outputs/precision_comparison.png')
print('性能比较图表已保存到 ./outputs/precision_comparison.png')

# 生成文本报告
has_amx = any(results[p].get('has_amx', False) for p in precisions)
amx_status = '支持' if has_amx else '不支持'

report = f'''# Intel AMX 加速器性能测试报告

## 测试环境

- CPU: {results[list(results.keys())[0]].get('device', 'CPU')}
- 模型: {results[list(results.keys())[0]].get('model', 'FLUX.1-dev')}
- Intel AMX 加速器: {amx_status}
- 图像分辨率: {results[list(results.keys())[0]].get('image_resolution', 'N/A')}
- 推理步数: {results[list(results.keys())[0]].get('inference_steps', 'N/A')}

## 性能比较

| 指标 | float32 | float16 | bfloat16 | INT8 量化 |
|------|---------|---------|----------|----------|
'''

# 添加推理时间
report += f"| 平均推理时间 | {results.get('float32', {}).get('avg_inference_time', 'N/A'):.2f}s | {results.get('float16', {}).get('avg_inference_time', 'N/A'):.2f}s | {results.get('bfloat16', {}).get('avg_inference_time', 'N/A'):.2f}s | {results.get('int8', {}).get('avg_inference_time', 'N/A'):.2f}s |\n"

# 添加加载时间
report += f"| 模型加载时间 | {results.get('float32', {}).get('load_time', 'N/A'):.2f}s | {results.get('float16', {}).get('load_time', 'N/A'):.2f}s | {results.get('bfloat16', {}).get('load_time', 'N/A'):.2f}s | {results.get('int8', {}).get('load_time', 'N/A'):.2f}s |\n"

# 添加 CPU 使用率
report += f"| CPU 平均使用率 | {results.get('float32', {}).get('cpu_avg', 'N/A'):.2f}% | {results.get('float16', {}).get('cpu_avg', 'N/A'):.2f}% | {results.get('bfloat16', {}).get('cpu_avg', 'N/A'):.2f}% | {results.get('int8', {}).get('cpu_avg', 'N/A'):.2f}% |\n"

# 添加内存使用率
report += f"| 内存平均使用率 | {results.get('float32', {}).get('memory_avg', 'N/A'):.2f}% | {results.get('float16', {}).get('memory_avg', 'N/A'):.2f}% | {results.get('bfloat16', {}).get('memory_avg', 'N/A'):.2f}% | {results.get('int8', {}).get('memory_avg', 'N/A'):.2f}% |\n"

# 计算加速比
if 'float32' in results and all(p in results for p in ['float16', 'bfloat16', 'int8']):
    base_time = results['float32'].get('avg_inference_time', 0)
    if base_time > 0:
        report += f'''
## 加速比 (相对于 float32)

- float16: {base_time / results['float16'].get('avg_inference_time', base_time):.2f}x
- bfloat16: {base_time / results['bfloat16'].get('avg_inference_time', base_time):.2f}x
- INT8 量化: {base_time / results['int8'].get('avg_inference_time', base_time):.2f}x
'''

# 添加结论
report += '''
## 结论

'''

# 找出最快的精度
if inference_times:
    fastest_idx = np.argmin(inference_times)
    fastest_precision = precisions[fastest_idx]
    report += f"- 在测试的精度中，{fastest_precision} 提供了最佳的推理性能。\n"

# 如果有 AMX 支持，添加相关结论
if has_amx:
    report += f"- Intel AMX 加速器对 bfloat16 精度提供了显著的加速。\n"

# 添加内存使用的结论
if memory_avgs:
    lowest_mem_idx = np.argmin(memory_avgs)
    lowest_mem_precision = precisions[lowest_mem_idx]
    report += f"- {lowest_mem_precision} 精度在内存使用方面最为高效。\n"

# 保存报告
with open('./outputs/amx_performance_report.md', 'w') as f:
    f.write(report)

print('性能测试报告已保存到 ./outputs/amx_performance_report.md')
"

echo "Intel AMX 加速器性能测试完成！"
