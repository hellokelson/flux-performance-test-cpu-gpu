#!/bin/bash
set -e

# 获取GPU实例IP
if [ ! -f gpu_instance_ip.txt ]; then
    echo "错误: 找不到GPU实例IP文件。请先运行launch_instances.sh"
    exit 1
fi

GPU_IP=$(cat gpu_instance_ip.txt)
PEM_PATH="/Users/zhangkap/aws/doc/authentication/ssh/zk-us-west-2.pem"

# 检查SSH连接
echo "检查与GPU实例的SSH连接..."
if ! ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$GPU_IP "echo SSH连接成功" &> /dev/null; then
    echo "错误: 无法通过SSH连接到GPU实例。请检查实例状态和安全组设置。"
    exit 1
fi

# 创建GPU实例配置脚本
cat > gpu_instance_setup.sh << 'EOF'
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
EOF

# 将脚本传输到GPU实例
echo "将脚本传输到GPU实例..."
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no gpu_instance_setup.sh ec2-user@$GPU_IP:~/
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no gpu_test_script.sh ec2-user@$GPU_IP:~/

# 执行设置脚本
echo "执行GPU实例设置脚本..."
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$GPU_IP "chmod +x gpu_instance_setup.sh && ./gpu_instance_setup.sh"
SETUP_EXIT_CODE=$?

# 如果设置脚本返回42，表示需要重启
if [ $SETUP_EXIT_CODE -eq 42 ]; then
    echo "需要重启GPU实例以应用NVIDIA驱动更改..."
    ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$GPU_IP "sudo reboot"
    
    # 等待实例重启
    echo "等待GPU实例重启..."
    sleep 60
    
    # 等待SSH可用
    echo "等待SSH连接恢复..."
    while ! ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ec2-user@$GPU_IP "echo SSH连接成功" &> /dev/null; do
        echo "等待SSH连接..."
        sleep 10
    done
    
    # 验证NVIDIA驱动安装
    echo "验证NVIDIA驱动安装..."
    ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$GPU_IP "nvidia-smi"
elif [ $SETUP_EXIT_CODE -ne 0 ]; then
    echo "错误: GPU实例设置脚本执行失败"
    exit 1
fi

# 执行测试脚本
echo "执行GPU测试脚本..."
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$GPU_IP "chmod +x gpu_test_script.sh && ./gpu_test_script.sh"

# 检查测试脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "错误: GPU测试脚本执行失败"
    exit 1
fi

# 获取测试结果
echo "获取测试结果..."
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$GPU_IP:~/flux_test/gpu_test_results.json ./

# 检查结果文件是否存在
if [ ! -f gpu_test_results.json ]; then
    echo "错误: 无法获取GPU测试结果文件"
    exit 1
fi

echo "GPU测试完成！"
