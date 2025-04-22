#!/bin/bash
set -e

echo "FLUX.1-dev 模型 GPU 性能测试一键脚本"

# 1. 设置环境
echo "步骤 1: 设置 ComfyUI 环境..."
bash setup_comfyui.sh

# 2. 部署模型
echo "步骤 2: 部署 FLUX.1-dev 模型..."
bash deploy_comfyui.sh

# 3. 运行测试
echo "步骤 3: 运行性能测试..."
bash run_comfyui_tests.sh

echo "全部步骤已完成！测试结果保存在 ./outputs 目录中"
