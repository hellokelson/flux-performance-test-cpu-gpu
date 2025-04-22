#!/bin/bash
set -e

echo "正在初始化 CPU (M7i) 环境..."

# 更新系统并安装基础依赖 (Amazon Linux 2023 使用 dnf)
sudo dnf update -y
sudo dnf install -y python3-pip python3-devel git wget htop

# 删除旧的虚拟环境（如果存在）
rm -rf flux_env

# 创建虚拟环境
python3 -m venv flux_env
source flux_env/bin/activate

# 安装 PyTorch (CPU 版本)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 安装兼容的依赖项
pip install huggingface_hub==0.16.4
pip install diffusers==0.21.4
pip install transformers==4.30.2
pip install accelerate==0.21.0
pip install safetensors==0.3.1
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3
pip install tokenizers==0.13.3

# 安装性能监控工具
pip install psutil py-cpuinfo

# 安装其他依赖
pip install pillow matplotlib numpy pandas

echo "CPU 环境初始化完成！"
