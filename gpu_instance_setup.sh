#!/bin/bash
set -e

echo "===== 开始配置GPU环境 ====="

# 更新系统
echo "更新系统..."
sudo dnf check-release-update
sudo dnf update -y

# 安装基本依赖
echo "安装基本依赖..."
sudo dnf install -y git python3-pip python3-devel gcc gcc-c++ make cmake wget

# 检查NVIDIA驱动是否已安装
echo "检查NVIDIA驱动是否已安装..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA驱动已安装，跳过NVIDIA驱动和CUDA安装"
    nvidia-smi
else
    echo "NVIDIA驱动未安装，开始安装NVIDIA驱动和CUDA..."
    
    # 安装DKMS
    echo "安装DKMS..."
    sudo dnf install -y dkms
    sudo systemctl enable --now dkms
    
    # 安装内核开发包和额外模块
    echo "安装内核开发包和额外模块..."
    if (uname -r | grep -q ^6.12.); then
      sudo dnf install -y kernel-devel-$(uname -r) kernel6.12-modules-extra
    else
      sudo dnf install -y kernel-devel-$(uname -r) kernel-modules-extra
    fi
    
    # 升级到最新版本
    echo "升级到最新版本..."
    sudo dnf upgrade --releasever=latest -y
    
    # 安装NVIDIA驱动和CUDA工具包
    echo "安装NVIDIA驱动和CUDA工具包..."
    sudo dnf install -y nvidia-release
    sudo dnf install -y nvidia-driver
    sudo dnf install -y cuda-toolkit
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    
    echo "NVIDIA驱动和CUDA安装完成，需要重启实例以应用更改"
    NEED_REBOOT=1
fi

# 安装Python依赖
echo "安装Python依赖..."
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio
python3 -m pip install numpy pillow requests huggingface_hub diffusers transformers accelerate safetensors protobuf sentencepiece

# 创建测试目录
echo "创建测试目录..."
mkdir -p ~/flux_test

echo "===== GPU环境配置完成 ====="

# 如果需要重启，则返回特定的退出码
if [ -n "$NEED_REBOOT" ]; then
    exit 42
fi
