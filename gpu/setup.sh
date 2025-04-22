#!/bin/bash
set -e

echo "正在初始化 GPU (G6) 环境..."

# 更新系统并安装基础依赖 (Amazon Linux 2023 使用 dnf)
sudo dnf update -y
sudo dnf install -y python3-pip python3-devel git wget htop
sudo dnf install kernel-modules-extra.x86_64

# 检查 NVIDIA 驱动是否已安装
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA 驱动未找到，开始安装..."
    
    # 按照 AWS 官方文档安装 NVIDIA 驱动
    # 参考: https://repost.aws/articles/ARwfQMxiC-QMOgWykD9mco1w/how-do-i-install-nvidia-gpu-driver-cuda-toolkit-nvidia-container-toolkit-on-amazon-ec2-instances-running-amazon-linux-2023-al2023
    
    # 安装 NVIDIA 存储库
    sudo dnf install -y nvidia-release
    
    # 安装最新的 NVIDIA 驱动 (DKMS 版本)
    sudo dnf module install -y nvidia-driver:latest-dkms
    
    # 检查安装是否成功
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA 驱动安装成功！"
        nvidia-smi
    else
        echo "NVIDIA 驱动安装失败。请检查错误信息。"
        exit 1
    fi
else
    echo "NVIDIA 驱动已安装，版本信息:"
    nvidia-smi
fi

# 检查 CUDA 是否已安装
if ! command -v nvcc &> /dev/null; then
    echo "CUDA 未找到，开始安装 CUDA 工具包..."
    
    # 安装 CUDA 工具包
    sudo dnf install -y cuda-toolkit
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # 立即应用环境变量
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    
    # 检查 CUDA 安装
    if command -v nvcc &> /dev/null; then
        echo "CUDA 工具包安装成功！"
        nvcc --version
    else
        echo "CUDA 工具包安装可能不完整，但我们将继续设置环境。"
    fi
else
    echo "CUDA 已安装，版本信息:"
    nvcc --version
fi

# 删除旧的虚拟环境（如果存在）
rm -rf flux_env

# 创建虚拟环境
python3 -m venv flux_env
source flux_env/bin/activate

# 安装 PyTorch (GPU 版本)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

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
pip install psutil py-cpuinfo nvidia-ml-py

# 安装其他依赖
pip install pillow matplotlib numpy pandas

# 验证 GPU 可用性
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count()); print('GPU 型号:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "GPU 环境初始化完成！"
