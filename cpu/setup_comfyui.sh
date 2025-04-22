#!/bin/bash
set -e

echo "正在初始化 CPU 环境并安装 ComfyUI..."

# 更新系统并安装基础依赖 (Amazon Linux 2023 使用 dnf)
if command -v dnf &> /dev/null; then
    sudo dnf update -y
    sudo dnf install -y python3-pip python3-devel git wget htop
elif command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev git wget htop
elif command -v brew &> /dev/null; then
    brew update
    brew install python3 git wget htop
else
    echo "未找到支持的包管理器，请手动安装依赖"
fi

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

# 安装 PyTorch (CPU 版本)
echo "安装 PyTorch CPU 版本..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 先安装特定版本的依赖项，解决兼容性问题
echo "安装特定版本的依赖项..."
pip install protobuf==3.20.3 sentencepiece==0.1.99
# 安装与 transformers 兼容的 huggingface-hub 版本
pip install huggingface-hub>=0.30.0

# 安装 ComfyUI 依赖
echo "安装 ComfyUI 依赖..."
# ComfyUI 的 requirements.txt 文件在克隆的仓库中
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "未找到 requirements.txt，安装常用依赖..."
    pip install numpy>=1.25.0 einops transformers>=4.28.1 tokenizers>=0.13.3
    pip install safetensors diffusers accelerate
    pip install opencv-python pillow scipy
fi

# 安装性能监控工具
echo "安装性能监控工具..."
pip install psutil py-cpuinfo matplotlib numpy pandas requests

# 创建模型目录
echo "创建模型目录..."
mkdir -p models/checkpoints

# 返回到原始目录
cd ../../

# 创建符号链接以便于访问
ln -sf comfyui/ComfyUI/models models

echo "ComfyUI 环境初始化完成！"
echo "使用以下命令激活环境："
echo "cd comfyui/ComfyUI && source venv/bin/activate"
