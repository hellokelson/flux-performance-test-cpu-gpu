#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用工具函数模块
"""

import os
import json
import time
import psutil
import platform
import cpuinfo
from PIL import Image

def get_cpu_info():
    """获取 CPU 信息"""
    try:
        info = cpuinfo.get_cpu_info()
        return info
    except Exception as e:
        print(f"获取 CPU 信息失败: {e}")
        return {
            "brand_raw": platform.processor(),
            "count": psutil.cpu_count(logical=True)
        }

def get_memory_info():
    """获取内存信息"""
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "percent": mem.percent,
        "used": mem.used,
        "free": mem.free
    }

def log_metrics(metrics, output_file):
    """记录性能指标到 JSON 文件"""
    # 添加时间戳
    metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 如果文件已存在，读取现有数据并追加
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    data.append(metrics)
                else:
                    data = [data, metrics]
        except:
            data = [metrics]
    else:
        data = [metrics]
    
    # 写入文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"性能指标已保存到 {output_file}")

def save_image(image, path):
    """保存图像到指定路径"""
    if isinstance(image, Image.Image):
        image.save(path)
        print(f"图像已保存到 {path}")
    else:
        print(f"无法保存图像: 不是有效的 PIL Image 对象")

def format_size(size_bytes):
    """格式化字节大小为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
