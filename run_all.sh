#!/bin/bash
set -e

echo "===== FLUX.1-dev 模型性能测试 ====="

# 启动实例
echo "步骤 1: 启动 EC2 实例..."
./launch_instances.sh

# 等待实例启动
echo "等待实例完全启动..."
sleep 60

# 配置 CPU 实例并执行测试
echo "步骤 2: 配置 CPU 实例环境并执行测试..."
./cpu_setup.sh

# 配置 GPU 实例并执行测试
echo "步骤 3: 配置 GPU 实例环境并执行测试..."
./gpu_setup.sh

# 分析测试结果
echo "步骤 4: 分析测试结果..."
python3 analyze_results.py

# 询问是否清理资源
echo ""
echo "测试完成！结果已保存到 cpu_test_results.json, gpu_test_results.json 和 performance_summary.json"
echo "性能对比图表已保存到 flux_performance_comparison.png"
echo ""
read -p "是否要清理资源（终止 EC2 实例）？(y/n): " answer

if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "步骤 5: 清理资源..."
    ./cleanup.sh
    echo "资源已清理"
else
    echo "跳过资源清理，请记得手动终止实例以避免不必要的费用"
fi

echo "===== 测试流程完成 ====="
