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

# 检查模型文件是否存在
if [ ! -f "comfyui/ComfyUI/models/checkpoints/flux_1_dev.safetensors" ]; then
    echo "警告: 模型文件不存在，尝试查找其他模型..."
    ls -la comfyui/ComfyUI/models/checkpoints/
    
    # 尝试查找其他模型
    MODEL_FILES=$(find comfyui/ComfyUI/models/checkpoints/ -name "*.safetensors" -o -name "*.ckpt" | head -1)
    if [ -n "$MODEL_FILES" ]; then
        echo "找到模型文件: $MODEL_FILES"
        # 创建符号链接
        ln -sf "$MODEL_FILES" comfyui/ComfyUI/models/checkpoints/flux_1_dev.safetensors
        echo "已创建符号链接: comfyui/ComfyUI/models/checkpoints/flux_1_dev.safetensors -> $MODEL_FILES"
    else
        echo "错误: 未找到任何模型文件，请先下载模型"
        exit 1
    fi
fi

# 启动 ComfyUI 服务器
echo "启动 ComfyUI 服务器..."
cd comfyui/ComfyUI
python main.py --cpu --port 8188 > comfyui_server.log 2>&1 &
SERVER_PID=$!
cd ../../

# 等待服务器启动
echo "等待 ComfyUI 服务器启动..."
for i in {1..60}; do
    if curl -s http://localhost:8188/ > /dev/null; then
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
curl -s http://localhost:8188/api/system-stats || echo "API 不可用"

# 等待 API 完全初始化
echo "等待 API 完全初始化..."
sleep 5

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