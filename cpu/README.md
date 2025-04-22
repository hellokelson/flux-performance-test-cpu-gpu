# FLUX.1-dev 模型 CPU 性能测试 (ComfyUI)

本目录包含使用 ComfyUI 测试 FLUX.1-dev 模型在 CPU 环境下的性能表现的脚本，特别关注 Intel AMX 加速器的性能提升。

## 文件说明

- `setup_comfyui.sh`: 安装 ComfyUI 和所需依赖
- `deploy_comfyui.sh`: 下载 FLUX.1-dev 模型
- `test_comfyui.py`: 使用 ComfyUI API 进行性能测试
- `run_comfyui_tests.sh`: 自动测试不同精度并生成比较报告

## 使用方法

### 1. 安装 ComfyUI 和依赖

```bash
bash setup_comfyui.sh
```

这将：
- 安装所需的系统依赖
- 克隆 ComfyUI 仓库
- 创建虚拟环境
- 安装 PyTorch CPU 版本和其他依赖

### 2. 下载 FLUX.1-dev 模型

```bash
bash deploy_comfyui.sh
```

这将从 Hugging Face 下载 FLUX.1-dev 模型到 ComfyUI 的模型目录。

### 3. 运行性能测试

#### 单次测试

```bash
# 启动 ComfyUI 服务器
cd comfyui/ComfyUI
source venv/bin/activate
python main.py --cpu --port 8188 &

# 在另一个终端中运行测试
python test_comfyui.py --precision half --output_dir ./outputs/float16
```

参数说明：
- `--precision`: 选择模型精度，可选值为 "full"（float32）或 "half"（float16）
- `--steps`: 设置推理步数
- `--prompt`: 自定义生成图像的提示词
- `--output_dir`: 指定输出目录

#### 自动测试

```bash
bash run_comfyui_tests.sh
```

这将：
1. 启动 ComfyUI 服务器
2. 测试 float32（full）和 float16（half）精度
3. 生成性能比较图表和报告
4. 关闭 ComfyUI 服务器

## 测试结果

测试结果将保存在 `./outputs/` 目录下，包括：
- 生成的图像
- 性能指标 JSON 文件
- 比较图表 (`comfyui_precision_comparison.png`)
- 性能测试报告 (`comfyui_amx_performance_report.md`)

## 手动使用 ComfyUI

如果您想直接通过 ComfyUI 的 Web 界面进行测试：

```bash
cd comfyui/ComfyUI
source venv/bin/activate
python main.py --cpu
```

然后在浏览器中访问 http://localhost:8188

## 注意事项

- ComfyUI 目前不直接支持 bfloat16 精度，因此我们只测试 float32 和 float16
- 确保您的系统有足够的内存运行 FLUX.1-dev 模型
- 测试结果可能因系统配置和负载而异
