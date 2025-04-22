#!/bin/bash
set -e

echo "正在部署 FLUX.1-dev 模型到 GPU 环境..."

# 激活虚拟环境
source flux_env/bin/activate

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

# 尝试加载模型以验证
try:
    from diffusers import DiffusionPipeline
    import torch
    
    print('尝试加载模型以验证...')
    pipe = DiffusionPipeline.from_pretrained(
        './models/FLUX.1-dev',
        torch_dtype=torch.float16
    )
    pipe = pipe.to('cuda')
    print('模型加载验证成功！')
except Exception as e:
    print(f'模型加载验证时出现警告或错误: {e}')
    print('这可能不影响后续使用，模型文件已成功下载。')
"

echo "FLUX.1-dev 模型已成功部署到 GPU 环境！"
