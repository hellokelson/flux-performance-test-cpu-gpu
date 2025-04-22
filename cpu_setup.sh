#!/bin/bash
set -e

# 获取CPU实例IP
if [ ! -f cpu_instance_ip.txt ]; then
    echo "错误: 找不到CPU实例IP文件。请先运行launch_instances.sh"
    exit 1
fi

CPU_IP=$(cat cpu_instance_ip.txt)
PEM_PATH="/Users/zhangkap/aws/doc/authentication/ssh/zk-us-west-2.pem"

# 检查SSH连接
echo "检查与CPU实例的SSH连接..."
if ! ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$CPU_IP "echo SSH连接成功" &> /dev/null; then
    echo "错误: 无法通过SSH连接到CPU实例。请检查实例状态和安全组设置。"
    exit 1
fi

# 创建CPU实例配置脚本
cat > cpu_instance_setup.sh << 'EOF'
#!/bin/bash
set -e

echo "===== 开始配置CPU环境 ====="

# 更新系统
echo "更新系统..."
sudo dnf update -y
sudo dnf install -y git python3-pip python3-devel gcc gcc-c++ make cmake wget

# 安装Python依赖
echo "安装Python依赖..."
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install numpy pillow requests huggingface_hub diffusers transformers accelerate safetensors

# 创建测试目录
echo "创建测试目录..."
mkdir -p ~/flux_test

# 启用AMX加速器
echo "启用AMX加速器..."
export PYTORCH_ENABLE_AMX=1

echo "===== CPU环境配置完成 ====="
EOF

# 将脚本传输到CPU实例
echo "将脚本传输到CPU实例..."
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no cpu_instance_setup.sh ec2-user@$CPU_IP:~/
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no cpu_test_script.sh ec2-user@$CPU_IP:~/

# 执行设置脚本
echo "执行CPU实例设置脚本..."
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$CPU_IP "chmod +x cpu_instance_setup.sh && ./cpu_instance_setup.sh"

# 检查设置脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "错误: CPU实例设置脚本执行失败"
    exit 1
fi

# 执行测试脚本
echo "执行CPU测试脚本..."
ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$CPU_IP "chmod +x cpu_test_script.sh && ./cpu_test_script.sh"

# 检查测试脚本是否成功执行
if [ $? -ne 0 ]; then
    echo "错误: CPU测试脚本执行失败"
    exit 1
fi

# 获取测试结果
echo "获取测试结果..."
scp -i "$PEM_PATH" -o StrictHostKeyChecking=no ec2-user@$CPU_IP:~/flux_test/cpu_test_results.json ./

# 检查结果文件是否存在
if [ ! -f cpu_test_results.json ]; then
    echo "错误: 无法获取CPU测试结果文件"
    exit 1
fi

echo "CPU测试完成！"
