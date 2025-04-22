#!/bin/bash
set -e

# 设置AWS区域
REGION="us-west-2"

# 设置实例类型
ARM_INSTANCE_TYPE="r8g.8xlarge"

# 设置AMI ID (Amazon Linux 2023 ARM)
ARM_AMI_ID="ami-0b0b7f0c98ff0d2a8"  # 请替换为最新的Amazon Linux 2023 ARM AMI ID

# 设置安全组ID
SECURITY_GROUP_ID="sg-0123456789abcdef"  # 请替换为您的安全组ID

# 设置密钥对名称
KEY_NAME="zk-us-west-2"  # 请替换为您的密钥对名称

# 设置子网ID
SUBNET_ID="subnet-0123456789abcdef"  # 请替换为您的子网ID

# 创建ARM实例
echo "创建ARM实例 (${ARM_INSTANCE_TYPE})..."
ARM_INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $ARM_AMI_ID \
    --instance-type $ARM_INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --subnet-id $SUBNET_ID \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=FLUX-ARM-Test}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "ARM实例ID: $ARM_INSTANCE_ID"

# 等待ARM实例运行
echo "等待ARM实例运行..."
aws ec2 wait instance-running --region $REGION --instance-ids $ARM_INSTANCE_ID

# 获取ARM实例公共IP
ARM_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $ARM_INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "ARM实例公共IP: $ARM_IP"
echo $ARM_IP > arm_instance_ip.txt

# 等待SSH可用
echo "等待SSH连接可用..."
while ! ssh -i "/Users/zhangkap/aws/doc/authentication/ssh/zk-us-west-2.pem" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ec2-user@$ARM_IP "echo SSH连接成功" &> /dev/null; do
    echo "等待SSH连接..."
    sleep 10
done

echo "ARM实例已准备就绪！"
echo "ARM实例IP: $ARM_IP"
