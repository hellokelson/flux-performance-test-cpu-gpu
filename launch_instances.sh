#!/bin/bash
set -e

# 设置AWS区域和其他参数
REGION="us-west-2"
VPC_ID="vpc-04a235b9b191b851f"
SUBNET_ID="subnet-001216118e2e41501"
KEY_NAME="zk-us-west-2"
AMI_ID="ami-05572e392e80aee89"
PEM_PATH="/Users/zhangkap/aws/doc/authentication/ssh/zk-us-west-2.pem"
SECURITY_GROUP_ID="sg-0675225b689027fa6"

# 检查AWS CLI是否可用
if ! command -v aws &> /dev/null; then
    echo "错误: AWS CLI 未安装或不在PATH中。请安装AWS CLI。"
    exit 1
fi

# 检查AWS凭证是否有效
echo "验证AWS凭证..."
if ! aws sts get-caller-identity --region $REGION &> /dev/null; then
    echo "错误: AWS凭证无效或未配置。请检查您的AWS配置。"
    exit 1
fi

# 检查是否已存在名为flux-cpu-test的实例
echo "检查是否已存在名为flux-cpu-test的实例..."
CPU_INSTANCE_ID=$(aws ec2 describe-instances \
  --region $REGION \
  --filters "Name=tag:Name,Values=flux-cpu-test" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query "Reservations[*].Instances[*].InstanceId" \
  --output text)

if [ -n "$CPU_INSTANCE_ID" ]; then
    echo "发现已存在的CPU实例 (ID: $CPU_INSTANCE_ID)"
    
    # 检查实例状态
    CPU_STATE=$(aws ec2 describe-instances \
      --region $REGION \
      --instance-ids $CPU_INSTANCE_ID \
      --query "Reservations[*].Instances[*].State.Name" \
      --output text)
    
    echo "CPU实例状态: $CPU_STATE"
    
    # 如果实例已停止，则启动它
    if [ "$CPU_STATE" == "stopped" ]; then
        echo "启动已停止的CPU实例..."
        aws ec2 start-instances --region $REGION --instance-ids $CPU_INSTANCE_ID
    fi
else
    # 创建块存储配置 - GP3, 600GB, 256MB/s吞吐量
    CPU_BLOCK_DEVICE_MAPPING='[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":600,"VolumeType":"gp3","DeleteOnTermination":true,"Throughput":256}}]'
    
    # 启动CPU实例 (M7i.8xlarge)
    echo "启动新的CPU实例 (M7i.8xlarge)..."
    CPU_INSTANCE_ID=$(aws ec2 run-instances \
      --region $REGION \
      --image-id $AMI_ID \
      --instance-type m7i.8xlarge \
      --key-name $KEY_NAME \
      --subnet-id $SUBNET_ID \
      --security-group-ids $SECURITY_GROUP_ID \
      --block-device-mappings "$CPU_BLOCK_DEVICE_MAPPING" \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=flux-cpu-test}]' \
      --count 1 \
      --query 'Instances[0].InstanceId' \
      --output text)

    # 检查CPU实例是否成功启动
    if [ -z "$CPU_INSTANCE_ID" ]; then
        echo "错误: 无法启动CPU实例。请检查您的AWS权限和配额。"
        exit 1
    fi
fi

echo "CPU实例ID: $CPU_INSTANCE_ID"

# 检查是否已存在名为flux-gpu-test的实例
echo "检查是否已存在名为flux-gpu-test的实例..."
GPU_INSTANCE_ID=$(aws ec2 describe-instances \
  --region $REGION \
  --filters "Name=tag:Name,Values=flux-gpu-test" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query "Reservations[*].Instances[*].InstanceId" \
  --output text)

if [ -n "$GPU_INSTANCE_ID" ]; then
    echo "发现已存在的GPU实例 (ID: $GPU_INSTANCE_ID)"
    
    # 检查实例状态
    GPU_STATE=$(aws ec2 describe-instances \
      --region $REGION \
      --instance-ids $GPU_INSTANCE_ID \
      --query "Reservations[*].Instances[*].State.Name" \
      --output text)
    
    echo "GPU实例状态: $GPU_STATE"
    
    # 如果实例已停止，则启动它
    if [ "$GPU_STATE" == "stopped" ]; then
        echo "启动已停止的GPU实例..."
        aws ec2 start-instances --region $REGION --instance-ids $GPU_INSTANCE_ID
    fi
else
    # 创建块存储配置 - GP3, 600GB, 256MB/s吞吐量
    GPU_BLOCK_DEVICE_MAPPING='[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":600,"VolumeType":"gp3","DeleteOnTermination":true,"Throughput":256}}]'
    
    # 启动GPU实例 (G6.4xlarge)
    echo "启动新的GPU实例 (G6.4xlarge)..."
    GPU_INSTANCE_ID=$(aws ec2 run-instances \
      --region $REGION \
      --image-id $AMI_ID \
      --instance-type g6.4xlarge \
      --key-name $KEY_NAME \
      --subnet-id $SUBNET_ID \
      --security-group-ids $SECURITY_GROUP_ID \
      --block-device-mappings "$GPU_BLOCK_DEVICE_MAPPING" \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=flux-gpu-test}]' \
      --count 1 \
      --query 'Instances[0].InstanceId' \
      --output text)

    # 检查GPU实例是否成功启动
    if [ -z "$GPU_INSTANCE_ID" ]; then
        echo "错误: 无法启动GPU实例。请检查您的AWS权限和配额。"
        echo "正在终止已启动的CPU实例..."
        aws ec2 terminate-instances --region $REGION --instance-ids $CPU_INSTANCE_ID
        exit 1
    fi
fi

echo "GPU实例ID: $GPU_INSTANCE_ID"

# 等待实例状态变为running
echo "等待实例启动..."
echo "等待CPU实例启动..."
if ! aws ec2 wait instance-running --region $REGION --instance-ids $CPU_INSTANCE_ID; then
    echo "错误: CPU实例未能成功启动。"
    echo "正在终止实例..."
    aws ec2 terminate-instances --region $REGION --instance-ids $CPU_INSTANCE_ID $GPU_INSTANCE_ID
    exit 1
fi

echo "等待GPU实例启动..."
if ! aws ec2 wait instance-running --region $REGION --instance-ids $GPU_INSTANCE_ID; then
    echo "错误: GPU实例未能成功启动。"
    echo "正在终止实例..."
    aws ec2 terminate-instances --region $REGION --instance-ids $CPU_INSTANCE_ID $GPU_INSTANCE_ID
    exit 1
fi

# 获取实例的公共IP地址
echo "获取实例公共IP地址..."
CPU_PUBLIC_IP=$(aws ec2 describe-instances \
  --region $REGION \
  --instance-ids $CPU_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

GPU_PUBLIC_IP=$(aws ec2 describe-instances \
  --region $REGION \
  --instance-ids $GPU_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

# 验证IP地址是否获取成功
if [ -z "$CPU_PUBLIC_IP" ] || [ "$CPU_PUBLIC_IP" == "None" ]; then
    echo "错误: 无法获取CPU实例的公共IP地址。"
    echo "正在终止实例..."
    aws ec2 terminate-instances --region $REGION --instance-ids $CPU_INSTANCE_ID $GPU_INSTANCE_ID
    exit 1
fi

if [ -z "$GPU_PUBLIC_IP" ] || [ "$GPU_PUBLIC_IP" == "None" ]; then
    echo "错误: 无法获取GPU实例的公共IP地址。"
    echo "正在终止实例..."
    aws ec2 terminate-instances --region $REGION --instance-ids $CPU_INSTANCE_ID $GPU_INSTANCE_ID
    exit 1
fi

echo "CPU实例公共IP: $CPU_PUBLIC_IP"
echo "GPU实例公共IP: $GPU_PUBLIC_IP"

# 将IP地址和实例ID保存到文件中，以便其他脚本使用
echo "$CPU_PUBLIC_IP" > cpu_instance_ip.txt
echo "$GPU_PUBLIC_IP" > gpu_instance_ip.txt
echo "$CPU_INSTANCE_ID" > cpu_instance_id.txt
echo "$GPU_INSTANCE_ID" > gpu_instance_id.txt

# 等待SSH可用
echo "等待SSH服务可用..."
CPU_SSH_READY=false
GPU_SSH_READY=false

for i in {1..30}; do
    if [ "$CPU_SSH_READY" != "true" ]; then
        if ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$CPU_PUBLIC_IP "echo SSH到CPU实例成功" &> /dev/null; then
            CPU_SSH_READY=true
            echo "CPU实例SSH已就绪"
        else
            echo "等待CPU实例SSH就绪... ($i/30)"
        fi
    fi
    
    if [ "$GPU_SSH_READY" != "true" ]; then
        if ssh -i "$PEM_PATH" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@$GPU_PUBLIC_IP "echo SSH到GPU实例成功" &> /dev/null; then
            GPU_SSH_READY=true
            echo "GPU实例SSH已就绪"
        else
            echo "等待GPU实例SSH就绪... ($i/30)"
        fi
    fi
    
    if [ "$CPU_SSH_READY" == "true" ] && [ "$GPU_SSH_READY" == "true" ]; then
        break
    fi
    
    sleep 10
done

if [ "$CPU_SSH_READY" != "true" ] || [ "$GPU_SSH_READY" != "true" ]; then
    echo "警告: 无法通过SSH连接到所有实例。请检查安全组设置。"
    echo "您可以继续尝试，但后续脚本可能会失败。"
fi

echo "实例启动完成！"
