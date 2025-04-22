#!/bin/bash
set -e

echo "正在初始化 GPU (G6) 环境..."

# 更新系统并安装基础依赖 (Amazon Linux 2023 使用 dnf)
sudo dnf update -y
sudo dnf install -y python3-pip python3-devel git wget htop

# 安装 NVIDIA 驱动和 CUDA 工具包
# 注意：G6 实例应该已经预装了 NVIDIA 驱动，但我们确保它已安装
if ! command -v nvidia-smi &> /dev/null; then
    echo "正在安装 NVIDIA 驱动..."
    # Amazon Linux 2023 安装 NVIDIA 驱动
    sudo dnf install -y kernel-devel-$(uname -r) gcc make
    sudo dnf install -y nvidia-driver nvidia-driver-cuda
fi

# 创建虚拟环境
python3 -m venv flux_env
source flux_env/bin/activate

# 安装 PyTorch (GPU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 FLUX 模型依赖
pip install diffusers transformers accelerate safetensors

# 安装性能监控工具
pip install psutil py-cpuinfo nvidia-ml-py

# 安装其他依赖
pip install pillow matplotlib numpy pandas

# 验证 GPU 可用性
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count()); print('GPU 型号:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "GPU 环境初始化完成！"
