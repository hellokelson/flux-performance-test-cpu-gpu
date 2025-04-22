#!/bin/bash
set -e

echo "正在部署 FLUX.1-dev 模型到 CPU 环境..."

# 激活虚拟环境
source flux_env/bin/activate

# 安装 huggingface_hub
pip install huggingface_hub

# 创建模型缓存目录
mkdir -p models/FLUX.1-dev

# 下载 FLUX.1-dev 模型
echo "正在下载 FLUX.1-dev 模型..."
python -c "
from huggingface_hub import snapshot_download
import time
import os

start_time = time.time()
print('开始下载模型...')

# 直接从 Hugging Face Hub 下载模型文件
model_path = snapshot_download(
    repo_id='black-forest-labs/FLUX.1-dev',
    local_dir='./models/FLUX.1-dev',
    local_dir_use_symlinks=False
)

end_time = time.time()
print(f'模型下载完成，耗时: {end_time - start_time:.2f} 秒')
print(f'模型已保存到: {model_path}')
"

echo "FLUX.1-dev 模型已成功部署到 CPU 环境！"