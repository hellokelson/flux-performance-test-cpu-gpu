#!/bin/bash
set -e

echo "正在部署 FLUX.1-dev 模型到 GPU 环境..."

# 激活虚拟环境
source flux_env/bin/activate

# 创建模型缓存目录
mkdir -p models

# 下载 FLUX.1-dev 模型
echo "正在下载 FLUX.1-dev 模型..."
python -c "
from diffusers import DiffusionPipeline
import torch
import time
import os

start_time = time.time()
print('开始下载模型...')

# 使用 GPU 加载模型
pipe = DiffusionPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    use_safetensors=True,
    torch_dtype=torch.float16,  # GPU 使用 float16
    use_fast_tokenizer=False  # 禁用 fast tokenizer 以避免兼容性问题
)

# 将模型移至 GPU
pipe = pipe.to('cuda')

# 保存模型到本地
pipe.save_pretrained('./models/FLUX.1-dev')

end_time = time.time()
print(f'模型下载和加载完成，耗时: {end_time - start_time:.2f} 秒')
"

echo "FLUX.1-dev 模型已成功部署到 GPU 环境！"
