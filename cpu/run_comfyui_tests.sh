#!/bin/bash
set -e

echo "开始 Intel AMX 加速器性能测试 (ComfyUI)..."

# 确保之前的 ComfyUI 服务器已关闭
pkill -f "python main.py --cpu" || true
sleep 2

# 激活虚拟环境
cd comfyui/ComfyUI
source venv/bin/activate
cd ../../

# 创建输出目录
mkdir -p outputs

# 启动 ComfyUI 服务器
echo "启动 ComfyUI 服务器..."
cd comfyui/ComfyUI
python main.py --cpu --port 8188 > comfyui_server.log 2>&1 &
SERVER_PID=$!
cd ../../

# 等待服务器启动
echo "等待 ComfyUI 服务器启动..."
for i in {1..60}; do
    if curl -s http://localhost:8188/api/system-stats > /dev/null; then
        echo "ComfyUI 服务器已启动"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ComfyUI 服务器启动超时"
        kill $SERVER_PID
        cat comfyui/ComfyUI/comfyui_server.log
        exit 1
    fi
    echo "等待中... ($i/60)"
    sleep 1
done

# 显示服务器日志
echo "ComfyUI 服务器日志:"
tail -n 20 comfyui/ComfyUI/comfyui_server.log

# 测试 API 是否可用
echo "测试 API 是否可用..."
curl -s http://localhost:8188/api/system-stats

# 测试不同精度
echo "测试 float32 精度 (full)..."
python test_comfyui.py --precision full --output_dir ./outputs/float32_comfyui

echo "测试 float16 精度 (half)..."
python test_comfyui.py --precision half --output_dir ./outputs/float16_comfyui

# 生成比较报告
echo "生成性能比较报告..."
# ... (保持原有代码)

# 关闭 ComfyUI 服务器
echo "关闭 ComfyUI 服务器..."
kill $SERVER_PID || true

echo "Intel AMX 加速器性能测试 (ComfyUI) 完成！"