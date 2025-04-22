#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUX.1-dev 模型 CPU 推理性能测试脚本 (ComfyUI)
专门用于测试 black-forest-labs/FLUX.1-dev 模型
支持 Intel AMX 加速器测试
"""

import os
import time
import argparse
import psutil
import json
import subprocess
import threading
import numpy as np
import requests
import sys
from PIL import Image
import base64
import io

def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.1-dev CPU 推理测试 (ComfyUI)")
    parser.add_argument("--prompt", type=str, default="一只可爱的小猫咪在草地上玩耍", help="生成图像的提示词")
    parser.add_argument("--negative_prompt", type=str, default="模糊的, 低质量的", help="负面提示词")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--height", type=int, default=512, help="图像高度")
    parser.add_argument("--width", type=int, default=512, help="图像宽度")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--precision", type=str, default="half", choices=["full", "half"], help="模型精度 (full=float32, half=float16)")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8188", help="ComfyUI 服务器 URL")
    return parser.parse_args()

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

def get_cpu_info():
    """获取 CPU 信息"""
    try:
        import py_cpuinfo
        info = py_cpuinfo.get_cpu_info()
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

def save_metrics(metrics, output_file):
    """保存性能指标"""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def start_comfyui_server(precision):
    """启动 ComfyUI 服务器"""
    print("启动 ComfyUI 服务器...")
    
    # 确保之前的 ComfyUI 服务器已关闭
    try:
        subprocess.run("pkill -f 'python main.py --cpu'", shell=True, check=False)
        time.sleep(2)
    except:
        pass
    
    # 构建启动命令
    cmd = "cd comfyui/ComfyUI && source venv/bin/activate && "
    
    # 根据精度设置环境变量
    if precision == "full":
        cmd += "python main.py --cpu --port 8188"
    else:  # half
        cmd += "python main.py --cpu --port 8188 --force-fp16"
    
    # 启动服务器进程
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 等待服务器启动
    print("等待 ComfyUI 服务器启动...")
    time.sleep(10)
    
    return process

def stop_comfyui_server(process):
    """停止 ComfyUI 服务器"""
    print("停止 ComfyUI 服务器...")
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

def wait_for_server(url, max_retries=10):
    """等待服务器准备就绪"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/system_stats")
            if response.status_code == 200:
                print("ComfyUI 服务器已准备就绪")
                return True
        except:
            pass
        
        print(f"等待 ComfyUI 服务器准备就绪... ({i+1}/{max_retries})")
        time.sleep(3)
    
    print("ComfyUI 服务器未响应")
    return False

def create_flux_workflow(prompt, negative_prompt, steps, width, height):
    """创建 FLUX.1-dev 工作流"""
    return {
        "3": {
            "inputs": {
                "seed": 123456789,
                "steps": steps,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler"
            }
        },
        "4": {
            "inputs": {
                "ckpt_name": "flux_1_dev.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "5": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
                "title": "Empty Latent Image"
            }
        },
        "6": {
            "inputs": {
                "text": prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Positive)"
            }
        },
        "7": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Negative)"
            }
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAE Decode"
            }
        },
        "9": {
            "inputs": {
                "filename_prefix": "flux_output",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "Save Image"
            }
        }
    }

def run_comfyui_inference(server_url, workflow, output_dir):
    """运行 ComfyUI 推理"""
    # 提交工作流
    print("提交工作流到 ComfyUI...")
    response = requests.post(f"{server_url}/prompt", json={"prompt": workflow})
    if response.status_code != 200:
        print(f"提交工作流失败: {response.text}")
        return None, 0
    
    prompt_id = response.json()["prompt_id"]
    print(f"工作流已提交，ID: {prompt_id}")
    
    # 等待工作流完成
    print("等待工作流完成...")
    start_time = time.time()
    
    while True:
        try:
            response = requests.get(f"{server_url}/history")
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history and history[prompt_id].get("outputs"):
                    # 找到图像节点的输出
                    for node_id, node_output in history[prompt_id]["outputs"].items():
                        if "images" in node_output:
                            image_data = node_output["images"][0]
                            image_filename = image_data["filename"]
                            inference_time = time.time() - start_time
                            print(f"工作流完成，耗时: {inference_time:.2f} 秒")
                            
                            # 下载图像
                            image_url = f"{server_url}/view?filename={image_filename}&subfolder=&type=temp"
                            image_response = requests.get(image_url)
                            if image_response.status_code == 200:
                                image = Image.open(io.BytesIO(image_response.content))
                                return image, inference_time
                            else:
                                print(f"下载图像失败: {image_response.status_code}")
                                return None, inference_time
            
            time.sleep(1)
        except Exception as e:
            print(f"检查工作流状态时出错: {e}")
            time.sleep(1)

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查 Intel AMX 支持
    has_amx = check_amx_support()
    
    # 记录 CPU 信息
    cpu_info = get_cpu_info()
    print(f"CPU 信息: {cpu_info['brand_raw']}, {cpu_info['count']} 核心")
    
    # 开始资源监控
    stop_monitor = threading.Event()
    resource_data = {'cpu_avg': 0, 'cpu_max': 0, 'memory_avg': 0, 'memory_max': 0}
    
    monitor_thread = threading.Thread(
        target=lambda: resource_data.update(monitor_resources(interval=0.5, stop_event=stop_monitor))
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    comfyui_process = None
    
    try:
        # 启动 ComfyUI 服务器
        print(f"开始测试 FLUX.1-dev 模型，精度: {args.precision}")
        load_start_time = time.time()
        
        comfyui_process = start_comfyui_server(args.precision)
        
        # 等待服务器准备就绪
        if not wait_for_server(args.server_url):
            raise Exception("ComfyUI 服务器未响应")
        
        load_time = time.time() - load_start_time
        print(f"ComfyUI 服务器启动完成，耗时: {load_time:.2f} 秒")
        
        # 创建工作流
        workflow = create_flux_workflow(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            width=args.width,
            height=args.height
        )
        
        # 执行推理
        print(f"开始生成图像，提示词: '{args.prompt}'")
        image, inference_time = run_comfyui_inference(args.server_url, workflow, args.output_dir)
        
        if image:
            # 保存图像
            image_path = os.path.join(args.output_dir, f"flux_image_{args.precision}.png")
            image.save(image_path)
            print(f"图像已保存到: {image_path}")
        
        # 停止资源监控
        stop_monitor.set()
        monitor_thread.join(timeout=1)
        
        # 记录性能指标
        metrics = {
            'device': 'CPU',
            'model': 'FLUX.1-dev',
            'precision': args.precision,
            'has_amx': has_amx,
            'load_time': load_time,
            'inference_time': inference_time,
            'cpu_avg': resource_data['cpu_avg'],
            'cpu_max': resource_data['cpu_max'],
            'memory_avg': resource_data['memory_avg'],
            'memory_max': resource_data['memory_max'],
            'image_resolution': f"{args.width}x{args.height}",
            'steps': args.steps
        }
        
        # 保存性能指标
        output_file = os.path.join(args.output_dir, f"flux_performance_{args.precision}_metrics.json")
        save_metrics(metrics, output_file)
        
        print("\n性能测试结果:")
        print(f"ComfyUI 启动时间: {load_time:.2f} 秒")
        print(f"推理时间: {inference_time:.2f} 秒")
        print(f"CPU 平均使用率: {resource_data['cpu_avg']:.2f}%")
        print(f"CPU 最大使用率: {resource_data['cpu_max']:.2f}%")
        print(f"内存平均使用率: {resource_data['memory_avg']:.2f}%")
        print(f"内存最大使用率: {resource_data['memory_max']:.2f}%")
        
        print(f"\n性能指标已保存到 {output_file}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保停止资源监控
        stop_monitor.set()
        
        # 停止 ComfyUI 服务器
        if comfyui_process:
            stop_comfyui_server(comfyui_process)

if __name__ == "__main__":
    main()