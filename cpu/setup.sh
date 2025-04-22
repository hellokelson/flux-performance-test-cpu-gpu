#!/bin/bash
set -e

echo "正在初始化 CPU (M7i) 环境..."

# 更新系统并安装基础依赖 (Amazon Linux 2023 使用 dnf)
sudo dnf update -y
sudo dnf install -y python3-pip python3-devel git wget htop

# 创建虚拟环境
python3 -m venv flux_env
source flux_env/bin/activate

# 安装 PyTorch (CPU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装缺失的依赖项
pip install sentencepiece protobuf

# 安装 FLUX 模型依赖
pip install diffusers transformers accelerate safetensors

# 安装性能监控工具
pip install psutil py-cpuinfo

# 安装其他依赖
pip install pillow matplotlib numpy pandas

echo "CPU 环境初始化完成！"
