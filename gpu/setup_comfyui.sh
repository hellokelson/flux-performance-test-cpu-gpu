#!/bin/bash
set -e

echo "正在初始化 GPU 环境并安装 ComfyUI..."

# 更新系统并安装基础依赖 (Amazon Linux 2023 使用 dnf)
sudo dnf update -y
sudo dnf install -y python3-pip python3-devel git wget htop

# 创建工作目录
mkdir -p comfyui
cd comfyui

# 克隆 ComfyUI 仓库
echo "克隆 ComfyUI 仓库..."
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装 PyTorch (GPU 版本)
echo "安装 PyTorch GPU 版本..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 ComfyUI 依赖
echo "安装 ComfyUI 依赖..."
pip install -r requirements.txt

# 安装性能监控工具
echo "安装性能监控工具..."
pip install psutil py-cpuinfo matplotlib numpy pandas pynvml

# 安装 xformers 以提高性能
echo "安装 xformers 以提高性能..."
pip install xformers

# 创建模型目录
echo "创建模型目录..."
mkdir -p models/checkpoints

# 返回到原始目录
cd ../../

# 创建符号链接以便于访问
ln -sf comfyui/ComfyUI/models models

echo "ComfyUI GPU 环境初始化完成！"
echo "使用以下命令激活环境："
echo "cd comfyui/ComfyUI && source venv/bin/activate"
