#!/bin/bash
set -e

echo "正在下载 FLUX.1-dev 模型到 ComfyUI..."

# 进入 ComfyUI 目录
cd comfyui/ComfyUI

# 激活虚拟环境
source venv/bin/activate

# 安装 huggingface_hub 用于下载模型
pip install huggingface_hub

# 列出仓库中的文件
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
from huggingface_hub import hf_hub_download
import os
import time
import shutil

start_time = time.time()
print('开始下载模型...')

# 创建模型目录
os.makedirs('models/checkpoints', exist_ok=True)

# 下载主模型文件 (使用正确的文件名)
try:
    # 尝试下载 model.safetensors
    model_path = hf_hub_download(
        repo_id='black-forest-labs/FLUX.1-dev',
        filename='model.safetensors',
        local_dir='models/checkpoints'
    )
    print(f'模型已下载到: {model_path}')
    
    # 重命名为 ComfyUI 可识别的名称
    target_path = os.path.join('models/checkpoints', 'flux_1_dev.safetensors')
    shutil.copy(model_path, target_path)
    print(f'模型已复制到: {target_path}')
except Exception as e:
    print(f'下载 model.safetensors 失败: {e}')
    print('尝试下载其他文件...')
    
    # 列出仓库中的所有 .safetensors 文件
    from huggingface_hub import list_repo_files
    files = list_repo_files('black-forest-labs/FLUX.1-dev')
    safetensors_files = [f for f in files if f.endswith('.safetensors')]
    
    if safetensors_files:
        print(f'找到以下 .safetensors 文件: {safetensors_files}')
        for file in safetensors_files:
            try:
                model_path = hf_hub_download(
                    repo_id='black-forest-labs/FLUX.1-dev',
                    filename=file,
                    local_dir='models/checkpoints'
                )
                print(f'文件已下载到: {model_path}')
                
                # 重命名为 ComfyUI 可识别的名称
                target_path = os.path.join('models/checkpoints', 'flux_1_dev.safetensors')
                shutil.copy(model_path, target_path)
                print(f'模型已复制到: {target_path}')
                break
            except Exception as e2:
                print(f'下载 {file} 失败: {e2}')
    else:
        print('未找到 .safetensors 文件')

# 下载配置文件
try:
    config_path = hf_hub_download(
        repo_id='black-forest-labs/FLUX.1-dev',
        filename='config.json',
        local_dir='models/checkpoints'
    )
    print(f'配置文件已下载到: {config_path}')
except Exception as e:
    print(f'下载配置文件失败: {e}')

end_time = time.time()
print(f'模型下载完成，耗时: {end_time - start_time:.2f} 秒')
"

echo "FLUX.1-dev 模型已成功部署到 ComfyUI！"
echo "您可以使用以下命令启动 ComfyUI："
echo "cd comfyui/ComfyUI && source venv/bin/activate && python main.py --cpu"