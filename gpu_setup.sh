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
EOF

# 将脚本传输到GPU实例
echo "将脚本传输到GPU实例..."
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no gpu_instance_setup.sh ec2-user@$GPU_IP:~/
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no gpu_test_script.sh ec2-user@$GPU_IP:~/

# 执行设置脚本
echo "执行GPU实例设置脚本..."
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$GPU_IP "chmod +x gpu_instance_setup.sh && ./gpu_instance_setup.sh"

# 检查设置脚本是否成功执行
if [ $? -ne 0 ]; then
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
