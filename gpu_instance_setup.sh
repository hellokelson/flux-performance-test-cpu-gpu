#!/bin/bash
set -e

echo "===== 开始配置GPU环境 ====="

# 更新系统
echo "更新系统..."
sudo dnf update -y
sudo dnf install -y git python3-pip python3-devel gcc gcc-c++ make cmake wget

# 安装NVIDIA驱动和CUDA
echo "安装NVIDIA驱动和CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    sudo dnf clean all
    sudo dnf -y module install nvidia-driver:latest-dkms
    sudo dnf -y install cuda
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
else
    echo "NVIDIA驱动已安装，跳过安装"
fi

# 验证NVIDIA驱动安装
echo "验证NVIDIA驱动安装..."
nvidia-smi

# 安装Python依赖
echo "安装Python依赖..."
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio
python3 -m pip install numpy pillow requests huggingface_hub diffusers transformers accelerate safetensors

# 创建测试目录
echo "创建测试目录..."
mkdir -p ~/flux_test

echo "===== GPU环境配置完成 ====="
