#!/bin/bash
set -e

echo "正在初始化 GPU (G6) 环境..."

# 更新系统并安装基础依赖 (Amazon Linux 2023 使用 dnf)
sudo dnf update -y
sudo dnf install -y python3-pip python3-devel git wget htop

# 检查 NVIDIA 驱动是否已安装
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA 驱动未找到，尝试安装..."
    
    # 在 Amazon Linux 2023 上，G6 实例应该已经预装了 NVIDIA 驱动
    # 如果没有，可以使用 NVIDIA 官方的驱动安装脚本
    echo "注意: G6 实例应该已经预装了 NVIDIA 驱动"
    echo "如果需要手动安装，请参考 AWS 文档: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html"
    
    # 检查是否有 CUDA 存储库
    if sudo dnf repolist | grep -q "cuda"; then
        echo "找到 CUDA 存储库，尝试安装 NVIDIA 驱动..."
        sudo dnf install -y cuda-drivers
    else
        echo "未找到 CUDA 存储库，尝试添加 NVIDIA 存储库..."
        # 添加 NVIDIA CUDA 存储库
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
        sudo dnf clean all
        sudo dnf install -y cuda-drivers
    fi
else
    echo "NVIDIA 驱动已安装，版本信息:"
    nvidia-smi
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
