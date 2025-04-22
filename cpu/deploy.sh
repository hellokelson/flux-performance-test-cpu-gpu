#!/bin/bash
set -e

echo "正在部署 FLUX.1-dev 模型到 CPU 环境..."

# 激活虚拟环境
source flux_env/bin/activate

# 创建模型缓存目录
mkdir -p models

# 下载 FLUX.1-dev 模型
echo "正在下载 FLUX.1-dev 模型..."
python -c "
from diffusers import StableDiffusionPipeline
import torch
import time
import os

start_time = time.time()
print('开始下载模型...')

try:
    # 尝试使用 StableDiffusionPipeline 加载模型
    pipe = StableDiffusionPipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        use_safetensors=True,
        torch_dtype=torch.float16,
        use_fast_tokenizer=False
    )
    
    # 将模型移至 CPU
    pipe = pipe.to('cpu')
    
    # 保存模型到本地
    pipe.save_pretrained('./models/FLUX.1-dev')
    
    print('模型加载成功并已保存到本地')
except Exception as e:
    print(f'使用 StableDiffusionPipeline 加载失败: {e}')
    
    # 尝试使用替代方法
    print('尝试使用替代方法...')
    
    # 使用 AutoPipelineForText2Image
    from diffusers import AutoPipelineForText2Image
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        use_safetensors=True,
        torch_dtype=torch.float16
    )
    
    # 将模型移至 CPU
    pipe = pipe.to('cpu')
    
    # 保存模型到本地
    pipe.save_pretrained('./models/FLUX.1-dev')

end_time = time.time()
print(f'模型下载和加载完成，耗时: {end_time - start_time:.2f} 秒')
"

echo "FLUX.1-dev 模型已成功部署到 CPU 环境！"