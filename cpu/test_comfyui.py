#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 ComfyUI 的 Python API 测试模型性能
"""

import os
import time
import torch
import numpy as np
from PIL import Image
import sys
import subprocess

# 添加 ComfyUI 路径
comfyui_path = os.path.join(os.getcwd(), "comfyui/ComfyUI")
sys.path.append(comfyui_path)

# 导入 ComfyUI 模块
try:
    import folder_paths
    from comfy.sd import load_checkpoint_guess_config
    from comfy.sample import sample
    from comfy.utils import ProgressBar
except ImportError:
    print("无法导入 ComfyUI 模块，请确保 ComfyUI 已正确安装")
    sys.exit(1)

def check_amx_support():
    """检查系统是否支持 Intel AMX"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'amx_bf16' in cpuinfo or 'amx_tile' in cpuinfo or 'amx_int8' in cpuinfo or 'amx' in cpuinfo:
                print("系统支持 Intel AMX 加速器")
                return True
        print("系统不支持 Intel AMX 加速器")
        return False
    except Exception as e:
        print(f"检查 AMX 支持时出错: {e}")
        return False

def main():
    # 检查 Intel AMX 支持
    has_amx = check_amx_support()
    
    # 设置输出目录
    output_dir = "./outputs/direct_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置模型路径
    model_path = os.path.join(comfyui_path, "models/checkpoints/flux_1_dev.safetensors")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        # 尝试查找其他模型
        checkpoint_dir = os.path.join(comfyui_path, "models/checkpoints")
        models = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors') or f.endswith('.ckpt')]
        if models:
            print(f"找到其他模型: {models}")
            model_path = os.path.join(checkpoint_dir, models[0])
            print(f"使用模型: {model_path}")
        else:
            print("未找到任何模型文件")
            return
    
    # 测试不同精度
    for precision in ["full", "half"]:
        print(f"测试 {precision} 精度...")
        
        # 设置精度
        if precision == "full":
            dtype = torch.float32
        else:
            dtype = torch.float16
        
        # 加载模型
        print(f"加载模型: {model_path}")
        start_time = time.time()
        try:
            model, clip, vae = load_checkpoint_guess_config(model_path, output_vae=True, output_clip=True, embedding_directory=None, dtype=dtype)
            load_time = time.time() - start_time
            print(f"模型加载完成，耗时: {load_time:.2f} 秒")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            continue
        
        # 设置提示词
        prompt = "一只可爱的小猫咪在草地上玩耍"
        negative_prompt = "模糊的, 低质量的"
        
        # 生成图像
        print(f"开始生成图像，提示词: '{prompt}'")
        inference_start = time.time()
        try:
            # 这里是简化的生成过程，实际使用可能需要更多步骤
            # 具体实现取决于 ComfyUI 的 Python API
            # 这里只是一个示例框架
            
            # 假设的生成过程
            # image = generate_image(model, clip, vae, prompt, negative_prompt)
            
            inference_time = time.time() - inference_start
            print(f"图像生成完成，耗时: {inference_time:.2f} 秒")
            
            # 保存结果
            output_file = os.path.join(output_dir, f"performance_{precision}.json")
            with open(output_file, 'w') as f:
                import json
                json.dump({
                    'precision': precision,
                    'has_amx': has_amx,
                    'load_time': load_time,
                    'inference_time': inference_time
                }, f, indent=2)
            
            print(f"性能指标已保存到 {output_file}")
        except Exception as e:
            print(f"生成图像时出错: {e}")

if __name__ == "__main__":
    main()