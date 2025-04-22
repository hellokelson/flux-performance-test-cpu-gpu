#!/bin/bash
set -e

echo "正在初始化 GPU (G6) 环境..."

# 更新系统并安装基础依赖 (Amazon Linux 2023 使用 dnf)
sudo dnf update -y
sudo dnf install -y python3-pip python3-devel git wget htop

# 检查 NVIDIA 驱动是否已安装
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA 驱动未找到。G6 实例应该已经预装了 NVIDIA 驱动。"
    echo "如果需要手动安装，请参考 AWS 文档: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html"
    
    # 对于 Amazon Linux 2023，可以尝试使用 NVIDIA 官方的 runfile 安装方法
    echo "尝试使用 NVIDIA 官方的 runfile 安装驱动..."
    
    # 安装必要的构建工具
    sudo dnf install -y gcc kernel-devel-$(uname -r) dkms
    
    # 下载最新的 NVIDIA 驱动
    DRIVER_VERSION="535.129.03"  # 可以根据需要更改版本
    wget https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run
    
    # 安装驱动
    sudo sh NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run --silent
    
    # 检查安装是否成功
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA 驱动安装成功！"
        nvidia-smi
    else
        echo "NVIDIA 驱动安装失败。请手动安装驱动。"
        echo "继续设置其他环境..."
    fi
else
    echo "NVIDIA 驱动已安装，版本信息:"
    nvidia-smi
fi

# 检查 CUDA 是否已安装
if ! command -v nvcc &> /dev/null; then
    echo "CUDA 未找到，尝试安装 CUDA 工具包..."
    
    # 对于 Amazon Linux 2023，可以尝试使用 NVIDIA 官方的 runfile 安装方法
    CUDA_VERSION="11.8.0"  # 可以根据需要更改版本
    CUDA_RUNFILE="cuda_${CUDA_VERSION}_520.61.05_linux.run"
    
    # 下载 CUDA runfile
    wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_RUNFILE}
    
    # 安装 CUDA，但不安装驱动（因为驱动应该已经安装）
    sudo sh ${CUDA_RUNFILE} --toolkit --silent --override
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # 立即应用环境变量
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    echo "CUDA 已安装，版本信息:"
    nvcc --version
fi

# 创建虚拟环境
python3 -m venv flux_env
source flux_env/bin/activate

# 安装 PyTorch (GPU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装缺失的依赖项
pip install sentencepiece protobuf

# 安装 FLUX 模型依赖
pip install diffusers transformers accelerate safetensors

# 安装性能监控工具
pip install psutil py-cpuinfo nvidia-ml-py

# 安装其他依赖
pip install pillow matplotlib numpy pandas

# 验证 GPU 可用性
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count()); print('GPU 型号:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "GPU 环境初始化完成！"
