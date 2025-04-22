#!/bin/bash
set -e

echo "正在下载 FLUX.1-dev 模型到 ComfyUI..."

# 进入 ComfyUI 目录
cd comfyui/ComfyUI

# 激活虚拟环境
source venv/bin/activate

# 安装 huggingface_hub 用于下载模型
pip install huggingface_hub

# 列出 FLUX.1-dev 仓库中的文件
echo "列出 FLUX.1-dev 仓库中的文件..."
python -c "
from huggingface_hub import list_repo_files
files = list_repo_files('black-forest-labs/FLUX.1-dev')
print('仓库中的文件:')
for file in files:
    print(f'- {file}')
"

# 下载 FLUX.1-dev 模型
echo "正在下载 FLUX.1-dev 模型..."
python -c "
from huggingface_hub import snapshot_download
import os
import time
import shutil

start_time = time.time()
print('开始下载模型...')

# 创建模型目录
os.makedirs('models/checkpoints', exist_ok=True)

# 下载整个仓库
model_path = snapshot_download(
    repo_id='black-forest-labs/FLUX.1-dev',
    local_dir='models/checkpoints/flux_1_dev_repo',
    local_dir_use_symlinks=False
)

print(f'模型已下载到: {model_path}')

# 复制主要模型文件到 checkpoints 目录
if os.path.exists(os.path.join(model_path, 'flux1-dev.safetensors')):
    shutil.copy(
        os.path.join(model_path, 'flux1-dev.safetensors'),
        'models/checkpoints/flux_1_dev.safetensors'
    )
    print(f'已复制模型文件到: models/checkpoints/flux_1_dev.safetensors')
else:
    print('警告: 未找到 flux1-dev.safetensors 文件')

# 检查并复制配置文件
config_files = [f for f in os.listdir(model_path) if f.endswith('.yaml')]
if config_files:
    for config_file in config_files:
        shutil.copy(
            os.path.join(model_path, config_file),
            os.path.join('models/checkpoints', config_file)
        )
        print(f'已复制配置文件: {config_file}')
else:
    print('警告: 未找到配置文件')

# 下载VAE
print('下载VAE模型...')
from huggingface_hub import hf_hub_download
vae_path = hf_hub_download(
    repo_id='stabilityai/sd-vae-ft-mse',
    filename='vae-ft-mse-840000-ema-pruned.safetensors',
    local_dir='models/vae'
)
print(f'VAE已下载到: {vae_path}')

end_time = time.time()
print(f'模型下载完成，耗时: {end_time - start_time:.2f} 秒')
"

echo "FLUX.1-dev 模型已成功部署到 ComfyUI！"
echo "您可以使用以下命令启动 ComfyUI："
echo "cd comfyui/ComfyUI && source venv/bin/activate && python main.py"
