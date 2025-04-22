#!/usr/bin/env python3
import os
import sys
import inspect
import torch

# Add ComfyUI path
comfyui_path = os.path.abspath("./comfyui/ComfyUI")
sys.path.append(comfyui_path)

try:
    # 导入必要的模块
    import folder_paths
    folder_paths.add_model_folder_path("checkpoints", os.path.join(comfyui_path, "models/checkpoints"))
    
    from comfy.sd import load_checkpoint_guess_config
    
    # Print the function signature
    print("Function signature:")
    print(inspect.signature(load_checkpoint_guess_config))
    
    # Print the function docstring
    print("\nFunction documentation:")
    print(load_checkpoint_guess_config.__doc__)
    
    # 检查模型文件
    model_path = os.path.join(comfyui_path, "models/checkpoints/flux_1_dev.safetensors")
    if os.path.exists(model_path):
        print(f"\n模型文件存在: {model_path}")
        
        # 尝试加载模型
        print("\n尝试加载模型...")
        result = load_checkpoint_guess_config(model_path)
        
        # 检查返回值
        print(f"\n返回值类型: {type(result)}")
        if isinstance(result, tuple):
            print(f"返回值长度: {len(result)}")
            for i, item in enumerate(result):
                print(f"返回值[{i}]类型: {type(item)}")
        
        # 尝试使用不同的参数
        print("\n尝试使用不同的参数加载模型...")
        try:
            result2 = load_checkpoint_guess_config(
                model_path,
                output_vae=True,
                output_clip=True
            )
            print(f"成功使用output_vae和output_clip参数")
            print(f"返回值类型: {type(result2)}")
            if isinstance(result2, tuple):
                print(f"返回值长度: {len(result2)}")
        except Exception as e:
            print(f"使用output_vae和output_clip参数失败: {e}")
        
        # 检查VAE
        print("\n检查VAE...")
        vae_path = os.path.join(comfyui_path, "models/vae")
        if os.path.exists(vae_path):
            print(f"VAE目录存在: {vae_path}")
            vae_files = os.listdir(vae_path)
            print(f"VAE文件: {vae_files}")
        else:
            print(f"VAE目录不存在: {vae_path}")
    else:
        print(f"\n模型文件不存在: {model_path}")
        
        # 查找其他可能的位置
        alt_paths = [
            "./models/FLUX.1-dev/flux1-dev.safetensors",
            "./models/flux_1_dev.safetensors",
            "./comfyui/ComfyUI/models/checkpoints/flux1-dev.safetensors",
            "./comfyui/ComfyUI/models/checkpoints/flux_1_dev_repo/flux1-dev.safetensors"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"找到替代模型文件: {alt_path}")
                break
        else:
            print("未找到任何模型文件")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()