#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUX.1-dev 模型 CPU 推理性能测试脚本 (基于 ComfyUI)
支持 Intel AMX 加速器测试
"""

import os
import time
import argparse
import psutil
import json
import subprocess
import requests
import threading
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1-dev ComfyUI CPU 推理测试")
    parser.add_argument("--prompt", type=str, default="一只可爱的小猫咪在草地上玩耍", help="生成图像的提示词")
    parser.add_argument("--negative_prompt", type=str, default="模糊的, 低质量的", help="负面提示词")
    parser.add_argument("--steps", type=int, default=30, help="推理步数")
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--precision", type=str, default="half", choices=["full", "half"], help="模型精度 (full=float32, half=float16)")
    parser.add_argument("--comfyui_dir", type=str, default="./comfyui/ComfyUI", help="ComfyUI 目录")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI 端口")
    parser.add_argument("--start_server", action="store_true", help="是否启动 ComfyUI 服务器")
    return parser.parse_args()

def check_amx_support():
    """检查系统是否支持 Intel AMX"""
    try:
        # 方法 1: 检查 /proc/cpuinfo 中是否有 amx 标志
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'amx_bf16' in cpuinfo or 'amx_tile' in cpuinfo or 'amx_int8' in cpuinfo or 'amx' in cpuinfo:
                print("系统支持 Intel AMX 加速器")
                return True
        
        # 方法 2: 使用 lscpu 命令
        try:
            result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, text=True)
            if 'amx_bf16' in result.stdout or 'amx_tile' in result.stdout or 'amx_int8' in result.stdout or 'amx' in result.stdout:
                print("系统支持 Intel AMX 加速器")
                return True
        except:
            pass
        
        # 方法 3: 检查实例类型
        try:
            result = subprocess.run(['curl', '-s', 'http://169.254.169.254/latest/meta-data/instance-type'], stdout=subprocess.PIPE, text=True)
            instance_type = result.stdout.strip()
            if instance_type.startswith('m7i'):
                print(f"检测到 {instance_type} 实例，支持 Intel AMX")
                return True
        except:
            pass
        
        print("系统不支持 Intel AMX 加速器")
        return False
    except Exception as e:
        print(f"检查 AMX 支持时出错: {e}")
        return False

def get_cpu_info():
    """获取 CPU 信息"""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return {
            'brand_raw': info.get('brand_raw', 'Unknown CPU'),
            'count': psutil.cpu_count(logical=True)
        }
    except:
        import platform
        return {
            'brand_raw': platform.processor(),
            'count': psutil.cpu_count(logical=True)
        }

def monitor_resources(interval=1.0, stop_event=None):
    """监控系统资源使用情况"""
    cpu_percentages = []
    memory_usages = []
    
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        cpu_percentages.append(cpu_percent)
        memory_usages.append(memory.percent)
        
        time.sleep(interval)
            
    return {
        'cpu_avg': np.mean(cpu_percentages) if cpu_percentages else 0,
        'cpu_max': np.max(cpu_percentages) if cpu_percentages else 0,
        'memory_avg': np.mean(memory_usages) if memory_usages else 0,
        'memory_max': np.max(memory_usages) if memory_usages else 0
    }

def start_comfyui_server(comfyui_dir, port=8188):
    """启动 ComfyUI 服务器"""
    print(f"启动 ComfyUI 服务器在端口 {port}...")
    
    # 激活虚拟环境
    venv_path = os.path.join(comfyui_dir, "venv")
    activate_script = os.path.join(venv_path, "bin", "activate")
    
    # 构建启动命令
    cmd = f"cd {comfyui_dir} && source {activate_script} && python main.py --cpu --port {port}"
    
    # 启动服务器
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 等待服务器启动
    print("等待 ComfyUI 服务器启动...")
    server_ready = False
    for _ in range(30):  # 最多等待 30 秒
        try:
            response = requests.get(f"http://127.0.0.1:{port}/")
            if response.status_code == 200:
                server_ready = True
                break
        except:
            pass
        time.sleep(1)
    
    if server_ready:
        print("ComfyUI 服务器已启动")
        return process
    else:
        process.terminate()
        raise Exception("ComfyUI 服务器启动失败")

def create_workflow(args):
    """创建 ComfyUI 工作流"""
    workflow = {
        "3": {
            "inputs": {
                "seed": 123456789,
                "steps": args.steps,
                "cfg": 7,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": "flux_1_dev.safetensors",
                "precision": args.precision  # "full" 或 "half"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": args.width,
                "height": args.height,
                "batch_size": args.batch_size
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": args.prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": args.negative_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "output",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }
    return workflow

def run_inference(args, port=8188):
    """运行推理并测量性能"""
    api_url = f"http://127.0.0.1:{port}/api/queue"
    
    # 创建工作流
    workflow = create_workflow(args)
    
    # 发送请求
    start_time = time.time()
    response = requests.post(api_url, json={"prompt": workflow})
    
    if response.status_code != 200:
        raise Exception(f"API 请求失败: {response.status_code}, {response.text}")
    
    prompt_id = response.json()["prompt_id"]
    
    # 等待完成
    while True:
        response = requests.get(f"http://127.0.0.1:{port}/api/queue")
        queue_status = response.json()
        
        if "queue_running" not in queue_status or prompt_id not in queue_status["queue_running"]:
            if "queue_pending" not in queue_status or prompt_id not in queue_status["queue_pending"]:
                break
        
        time.sleep(1)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 获取生成的图像
    history_url = f"http://127.0.0.1:{port}/api/history"
    response = requests.get(history_url)
    history = response.json()
    
    # 查找最新的输出
    latest_output = None
    if prompt_id in history:
        outputs = history[prompt_id]["outputs"]
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                latest_output = node_output["images"][0]["filename"]
                break
    
    return inference_time, latest_output

def save_metrics(metrics, output_file):
    """保存性能指标"""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查 Intel AMX 支持
    has_amx = check_amx_support()
    
    # 记录 CPU 信息
    cpu_info = get_cpu_info()
    print(f"CPU 信息: {cpu_info['brand_raw']}, {cpu_info['count']} 核心")
    
    # 启动 ComfyUI 服务器（如果需要）
    server_process = None
    if args.start_server:
        server_process = start_comfyui_server(args.comfyui_dir, args.port)
    
    # 开始资源监控
    stop_monitor = threading.Event()
    resource_data = {'cpu_avg': 0, 'cpu_max': 0, 'memory_avg': 0, 'memory_max': 0}
    
    monitor_thread = threading.Thread(
        target=lambda: resource_data.update(monitor_resources(interval=0.5, stop_event=stop_monitor))
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # 执行推理
        print(f"开始生成图像，提示词: '{args.prompt}'")
        inference_times = []
        output_images = []
        
        for i in range(args.batch_size):
            try:
                inference_time, output_image = run_inference(args, args.port)
                inference_times.append(inference_time)
                output_images.append(output_image)
                
                print(f"图像 {i+1}/{args.batch_size} 生成完成，耗时: {inference_time:.2f} 秒")
                
                # 复制生成的图像到输出目录
                if output_image:
                    src_path = os.path.join(args.comfyui_dir, "output", output_image)
                    dst_path = os.path.join(args.output_dir, f"image_{i+1}.png")
                    if os.path.exists(src_path):
                        import shutil
                        shutil.copy(src_path, dst_path)
                        print(f"图像已保存到 {dst_path}")
            except Exception as e:
                print(f"生成图像 {i+1}/{args.batch_size} 时出错: {e}")
        
        # 停止资源监控
        stop_monitor.set()
        monitor_thread.join(timeout=1)
        
        # 计算并打印性能指标
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            
            print("\n性能测试结果:")
            print(f"平均推理时间: {avg_inference_time:.2f} 秒/图像")
            print(f"CPU 平均使用率: {resource_data['cpu_avg']:.2f}%")
            print(f"CPU 最大使用率: {resource_data['cpu_max']:.2f}%")
            print(f"内存平均使用率: {resource_data['memory_avg']:.2f}%")
            print(f"内存最大使用率: {resource_data['memory_max']:.2f}%")
            
            # 记录性能指标
            metrics = {
                'device': 'CPU',
                'model': 'FLUX.1-dev',
                'precision': args.precision,
                'has_amx': has_amx,
                'avg_inference_time': avg_inference_time,
                'cpu_avg': resource_data['cpu_avg'],
                'cpu_max': resource_data['cpu_max'],
                'memory_avg': resource_data['memory_avg'],
                'memory_max': resource_data['memory_max'],
                'batch_size': args.batch_size,
                'image_resolution': f"{args.width}x{args.height}",
                'steps': args.steps
            }
            
            # 保存性能指标
            output_file = os.path.join(args.output_dir, f"comfyui_performance_{args.precision}_metrics.json")
            save_metrics(metrics, output_file)
            
            print(f"\n所有图像已保存到 {args.output_dir} 目录")
            print(f"性能指标已保存到 {output_file}")
        else:
            print("\n未能成功生成任何图像，无法计算性能指标。")
    
    finally:
        # 确保停止资源监控
        stop_monitor.set()
        
        # 关闭 ComfyUI 服务器（如果是我们启动的）
        if server_process:
            print("关闭 ComfyUI 服务器...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()
