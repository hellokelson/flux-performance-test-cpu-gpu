#!/bin/bash
set -e

echo "启动 ComfyUI 服务..."

# 进入 ComfyUI 目录
cd comfyui/ComfyUI

# 激活虚拟环境
source venv/bin/activate

# 启动 ComfyUI (GPU 模式)
python main.py --listen 0.0.0.0 --port 8188

echo "ComfyUI 服务已停止"
