#!/bin/bash
set -e

# 获取ARM实例IP
if [ ! -f arm_instance_ip.txt ]; then
    echo "错误: 找不到ARM实例IP文件。请先运行launch_arm_instance.sh"
    exit 1
fi

ARM_IP=$(cat arm_instance_ip.txt)
PEM_PATH="/Users/zhangkap/aws/doc/authentication/ssh/zk-us-west-2.pem"

# 检查SSH连接
echo "检查与ARM实例的SSH连接..."
if ! ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$ARM_IP "echo SSH连接成功" &> /dev/null; then
    echo "错误: 无法通过SSH连接到ARM实例。请检查实例状态和安全组设置。"
    exit 1
fi

# 创建ARM实例配置脚本
cat > arm_instance_setup.sh << 'EOF'
#!/bin/bash
set -e

echo "===== 开始配置ARM环境 ====="

# 更新系统
echo "更新系统..."
sudo dnf update -y

# 安装基本依赖
echo "安装基本依赖..."
sudo dnf install -y git python3-pip python3-devel gcc gcc-c++ make cmake wget

# 安装Python依赖
echo "安装Python依赖..."
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio
python3 -m pip install numpy pillow requests huggingface_hub diffusers transformers accelerate safetensors protobuf sentencepiece

# 创建测试目录
echo "创建测试目录..."
mkdir -p ~/flux_test

echo "===== ARM环境配置完成 ====="
EOF

# 将脚本传输到ARM实例
echo "将脚本传输到ARM实例..."
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no arm_instance_setup.sh ec2-user@$ARM_IP:~/
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no arm_cpu_test_script.sh ec2-user@$ARM_IP:~/

# 执行设置脚本
echo "执行ARM实例设置脚本..."
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$ARM_IP "chmod +x arm_instance_setup.sh && ./arm_instance_setup.sh"
SETUP_EXIT_CODE=$?

if [ $SETUP_EXIT_CODE -ne 0 ]; then
    echo "错误: ARM实例设置脚本执行失败"
    exit 1
fi

# 执行CPU测试脚本
echo "执行ARM CPU测试脚本..."
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$ARM_IP "chmod +x arm_cpu_test_script.sh && ./arm_cpu_test_script.sh"

# 检查CPU测试脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "错误: ARM CPU测试脚本执行失败"
    exit 1
fi

# 获取测试结果
echo "获取测试结果..."
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$ARM_IP:~/flux_test/arm_cpu_test_results.json ./

# 检查结果文件是否存在
if [ ! -f arm_cpu_test_results.json ]; then
    echo "错误: 无法获取ARM测试结果文件"
    exit 1
fi

echo "ARM测试完成！"
