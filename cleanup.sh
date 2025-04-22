#!/bin/bash
set -e

echo "开始清理资源..."

# 检查实例状态的函数
check_instance_status() {
    local instance_id=$1
    local status=$(aws ec2 describe-instances --instance-ids $instance_id --query 'Reservations[0].Instances[0].State.Name' --output text)
    echo $status
}

# 检查CPU实例ID文件是否存在
if [ -f cpu_instance_id.txt ]; then
    CPU_INSTANCE_ID=$(cat cpu_instance_id.txt)
    
    # 检查CPU实例状态
    CPU_STATUS=$(check_instance_status $CPU_INSTANCE_ID)
    echo "CPU实例 (ID: $CPU_INSTANCE_ID) 当前状态: $CPU_STATUS"
    
    if [[ "$CPU_STATUS" == "terminated" || "$CPU_STATUS" == "shutting-down" ]]; then
        echo "CPU实例已经终止或正在终止，跳过操作"
    elif [[ "$CPU_STATUS" == "stopped" ]]; then
        echo "CPU实例已经停止，直接执行终止操作..."
        
        # 关闭终止保护
        aws ec2 modify-instance-attribute --instance-id $CPU_INSTANCE_ID --no-disable-api-termination
        
        # 终止实例
        aws ec2 terminate-instances --instance-ids $CPU_INSTANCE_ID
        echo "CPU实例终止请求已发送"
    else
        echo "关闭CPU实例的关机保护和终止保护..."
        
        # 关闭关机保护
        aws ec2 modify-instance-attribute --instance-id $CPU_INSTANCE_ID --no-disable-api-stop
        
        # 关闭终止保护
        aws ec2 modify-instance-attribute --instance-id $CPU_INSTANCE_ID --no-disable-api-termination
        
        echo "终止CPU实例 (ID: $CPU_INSTANCE_ID)..."
        aws ec2 terminate-instances --instance-ids $CPU_INSTANCE_ID
        echo "CPU实例终止请求已发送"
    fi
else
    echo "警告: 找不到CPU实例ID文件，跳过终止CPU实例"
fi

# 检查GPU实例ID文件是否存在
if [ -f gpu_instance_id.txt ]; then
    GPU_INSTANCE_ID=$(cat gpu_instance_id.txt)
    
    # 检查GPU实例状态
    GPU_STATUS=$(check_instance_status $GPU_INSTANCE_ID)
    echo "GPU实例 (ID: $GPU_INSTANCE_ID) 当前状态: $GPU_STATUS"
    
    if [[ "$GPU_STATUS" == "terminated" || "$GPU_STATUS" == "shutting-down" ]]; then
        echo "GPU实例已经终止或正在终止，跳过操作"
    elif [[ "$GPU_STATUS" == "stopped" ]]; then
        echo "GPU实例已经停止，直接执行终止操作..."
        
        # 关闭终止保护
        aws ec2 modify-instance-attribute --instance-id $GPU_INSTANCE_ID --no-disable-api-termination
        
        # 终止实例
        aws ec2 terminate-instances --instance-ids $GPU_INSTANCE_ID
        echo "GPU实例终止请求已发送"
    else
        echo "关闭GPU实例的关机保护和终止保护..."
        
        # 关闭关机保护
        aws ec2 modify-instance-attribute --instance-id $GPU_INSTANCE_ID --no-disable-api-stop
        
        # 关闭终止保护
        aws ec2 modify-instance-attribute --instance-id $GPU_INSTANCE_ID --no-disable-api-termination
        
        echo "终止GPU实例 (ID: $GPU_INSTANCE_ID)..."
        aws ec2 terminate-instances --instance-ids $GPU_INSTANCE_ID
        echo "GPU实例终止请求已发送"
    fi
else
    echo "警告: 找不到GPU实例ID文件，跳过终止GPU实例"
fi

echo "资源清理完成"
