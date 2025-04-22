#!/bin/bash
set -e

echo "===== 开始配置CPU环境 ====="

# 更新系统
echo "更新系统..."
sudo dnf update -y
sudo dnf install -y git python3-pip python3-devel gcc gcc-c++ make cmake wget

# 安装Python依赖
echo "安装Python依赖..."
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install numpy pillow requests huggingface_hub diffusers transformers accelerate safetensors

# 创建测试目录
echo "创建测试目录..."
mkdir -p ~/flux_test

# 启用AMX加速器
echo "启用AMX加速器..."
export PYTORCH_ENABLE_AMX=1

echo "===== CPU环境配置完成 ====="
