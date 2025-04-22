#!/bin/bash
set -e

echo "===== FLUX.1-dev 模型 CPU 性能测试 (ComfyUI) ====="
echo "此脚本将执行完整的安装、部署和测试流程"

# 1. 安装 ComfyUI 和依赖
echo -e "\n\n===== 步骤 1: 安装 ComfyUI 和依赖 ====="
bash setup_comfyui.sh

# 2. 下载 FLUX.1-dev 模型
echo -e "\n\n===== 步骤 2: 下载 FLUX.1-dev 模型 ====="
bash deploy_comfyui.sh

# 3. 运行性能测试
echo -e "\n\n===== 步骤 3: 运行性能测试 ====="
bash run_comfyui_tests.sh

echo -e "\n\n===== 测试完成 ====="
echo "结果保存在 ./outputs/ 目录中"
echo "性能比较图表: ./outputs/comfyui_precision_comparison.png"
echo "性能测试报告: ./outputs/comfyui_amx_performance_report.md"
