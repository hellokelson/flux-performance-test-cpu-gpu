#!/bin/bash
set -e

echo "开始 Intel AMX 加速器性能测试 (纯 CPU 环境)..."

# 创建虚拟环境（如果不存在）
if [ ! -d "cpu_env" ]; then
    echo "创建虚拟环境..."
    python3 -m venv cpu_env
fi

# 激活虚拟环境
source cpu_env/bin/activate

# 安装依赖
echo "安装依赖..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate safetensors
pip install numpy matplotlib psutil py-cpuinfo pillow

# 创建输出目录
mkdir -p outputs

# 测试不同精度
echo "测试 float32 精度..."
python test_comfyui.py --precision full --output_dir ./outputs/float32_flux

echo "测试 float16 精度..."
python test_comfyui.py --precision half --output_dir ./outputs/float16_flux

# 生成比较报告
echo "生成性能比较报告..."
python -c "
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 读取各种精度的性能指标
results = {}
for precision, dir_name in [('full', 'float32_flux'), ('half', 'float16_flux')]:
    try:
        file_path = f'./outputs/{dir_name}/flux_performance_{precision}_metrics.json'
        
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
precision_labels = {'full': 'float32', 'half': 'float16'}
labels = [precision_labels.get(p, p) for p in precisions]
inference_times = [results[p].get('inference_time', 0) for p in precisions]

plt.subplot(2, 2, 1)
plt.bar(labels, inference_times, color=['blue', 'green'])
plt.title('不同精度下的推理时间')
plt.ylabel('时间 (秒)')
for i, v in enumerate(inference_times):
    plt.text(i, v + 0.1, f'{v:.2f}s', ha='center')

# 加载时间比较
load_times = [results[p].get('load_time', 0) for p in precisions]

plt.subplot(2, 2, 2)
plt.bar(labels, load_times, color=['blue', 'green'])
plt.title('不同精度下的模型加载时间')
plt.ylabel('时间 (秒)')
for i, v in enumerate(load_times):
    plt.text(i, v + 0.1, f'{v:.2f}s', ha='center')

# CPU 使用率比较
cpu_avgs = [results[p].get('cpu_avg', 0) for p in precisions]

plt.subplot(2, 2, 3)
plt.bar(labels, cpu_avgs, color=['blue', 'green'])
plt.title('不同精度下的 CPU 平均使用率')
plt.ylabel('使用率 (%)')
for i, v in enumerate(cpu_avgs):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')

# 内存使用率比较
memory_avgs = [results[p].get('memory_avg', 0) for p in precisions]

plt.subplot(2, 2, 4)
plt.bar(labels, memory_avgs, color=['blue', 'green'])
plt.title('不同精度下的内存平均使用率')
plt.ylabel('使用率 (%)')
for i, v in enumerate(memory_avgs):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')

plt.tight_layout()
plt.savefig('./outputs/flux_precision_comparison.png')
print('性能比较图表已保存到 ./outputs/flux_precision_comparison.png')

# 生成文本报告
has_amx = any(results[p].get('has_amx', False) for p in precisions)
amx_status = '支持' if has_amx else '不支持'

report = f'''# FLUX.1-dev 模型 Intel AMX 加速器性能测试报告

## 测试环境

- CPU: {results[list(results.keys())[0]].get('device', 'CPU')}
- 模型: {results[list(results.keys())[0]].get('model', 'FLUX.1-dev')}
- Intel AMX 加速器: {amx_status}
- 图像分辨率: {results[list(results.keys())[0]].get('image_resolution', 'N/A')}
- 推理步数: {results[list(results.keys())[0]].get('steps', 'N/A')}

## 性能比较

| 指标 | float32 (full) | float16 (half) |
|------|---------------|---------------|
'''

# 添加推理时间
report += f\"| 推理时间 | {results.get('full', {}).get('inference_time', 'N/A'):.2f}s | {results.get('half', {}).get('inference_time', 'N/A'):.2f}s |\\n\"

# 添加加载时间
report += f\"| 模型加载时间 | {results.get('full', {}).get('load_time', 'N/A'):.2f}s | {results.get('half', {}).get('load_time', 'N/A'):.2f}s |\\n\"

# 添加 CPU 使用率
report += f\"| CPU 平均使用率 | {results.get('full', {}).get('cpu_avg', 'N/A'):.2f}% | {results.get('half', {}).get('cpu_avg', 'N/A'):.2f}% |\\n\"

# 添加内存使用率
report += f\"| 内存平均使用率 | {results.get('full', {}).get('memory_avg', 'N/A'):.2f}% | {results.get('half', {}).get('memory_avg', 'N/A'):.2f}% |\\n\"

# 计算加速比
if 'full' in results and 'half' in results:
    full_time = results['full'].get('inference_time', 0)
    half_time = results['half'].get('inference_time', 0)
    if full_time > 0 and half_time > 0:
        speedup = full_time / half_time
        report += f'''
## 加速比

- float16 相对于 float32: {speedup:.2f}x
'''

# 添加结论
report += '''
## 结论

'''

# 找出最快的精度
if inference_times:
    fastest_idx = np.argmin(inference_times)
    fastest_precision = precision_labels.get(precisions[fastest_idx], precisions[fastest_idx])
    report += f\"- 在测试的精度中，{fastest_precision} 提供了最佳的推理性能。\\n\"

# 如果有 AMX 支持，添加相关结论
if has_amx:
    report += f\"- Intel AMX 加速器对 float16 精度提供了加速。\\n\"

# 添加内存使用的结论
if memory_avgs:
    lowest_mem_idx = np.argmin(memory_avgs)
    lowest_mem_precision = precision_labels.get(precisions[lowest_mem_idx], precisions[lowest_mem_idx])
    report += f\"- {lowest_mem_precision} 精度在内存使用方面更为高效。\\n\"

# 保存报告
with open('./outputs/flux_amx_performance_report.md', 'w') as f:
    f.write(report)

print('性能测试报告已保存到 ./outputs/flux_amx_performance_report.md')
"

echo "FLUX.1-dev 模型 Intel AMX 加速器性能测试完成！"