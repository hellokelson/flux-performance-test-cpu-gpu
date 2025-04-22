#!/bin/bash
set -e

echo "正在下载 FLUX.1-dev 模型到 ComfyUI..."

# 进入 ComfyUI 目录
cd comfyui/ComfyUI

# 激活虚拟环境
source venv/bin/activate

# 安装 huggingface_hub 用于下载模型
pip install huggingface_hub

# 下载 FLUX.1-dev 模型
echo "正在下载 FLUX.1-dev 模型..."
python -c "
from huggingface_hub import hf_hub_download
import os
import time

start_time = time.time()
print('开始下载模型...')

# 创建模型目录
os.makedirs('models/checkpoints', exist_ok=True)

# 下载主模型文件
model_path = hf_hub_download(
    repo_id='black-forest-labs/FLUX.1-dev',
    filename='flux_1_dev.safetensors',
    local_dir='models/checkpoints',
    local_dir_use_symlinks=False
)

print(f'模型已下载到: {model_path}')

# 下载配置文件
config_path = hf_hub_download(
    repo_id='black-forest-labs/FLUX.1-dev',
    filename='config.json',
    local_dir='models/checkpoints',
    local_dir_use_symlinks=False
)

print(f'配置文件已下载到: {config_path}')

end_time = time.time()
print(f'模型下载完成，耗时: {end_time - start_time:.2f} 秒')
"

echo "FLUX.1-dev 模型已成功部署到 ComfyUI！"
echo "您可以使用以下命令启动 ComfyUI："
echo "cd comfyui/ComfyUI && source venv/bin/activate && python main.py --cpu"
