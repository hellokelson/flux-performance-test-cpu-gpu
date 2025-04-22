#!/bin/bash
set -e

echo "启动 ComfyUI 服务器进行手动测试..."

# 确保之前的 ComfyUI 服务器已关闭
pkill -f "python main.py --cpu" || true
sleep 2

# 激活虚拟环境并启动 ComfyUI
cd comfyui/ComfyUI
source venv/bin/activate

# 启动 ComfyUI 服务器
echo "ComfyUI 服务器已启动，请在浏览器中访问 http://localhost:8188"
echo "按 Ctrl+C 停止服务器"
python main.py --cpu --port 8188