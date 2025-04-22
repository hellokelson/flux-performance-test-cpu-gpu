# FLUX.1-dev 模型 CPU 性能测试

本目录包含在 CPU 环境下测试 black-forest-labs/FLUX.1-dev 模型性能的脚本。测试使用 ComfyUI 作为推理框架。

## 文件说明

- `setup_comfyui.sh`: 安装 ComfyUI 和所需依赖
- `deploy_comfyui.sh`: 下载 FLUX.1-dev 模型到 ComfyUI
- `start_comfyui.sh`: 启动 ComfyUI 服务器进行手动测试
- `test_comfyui.py`: 自动化测试脚本，用于测量性能指标
- `run_comfyui_tests.sh`: 批量运行不同精度的测试并生成比较报告
- `all_in_one.sh`: 一键执行完整测试流程（安装、部署、测试）

## 使用方法

### 一键测试

执行以下命令完成全部测试流程：

```bash
bash all_in_one.sh
```

### 分步测试

1. 安装 ComfyUI 和依赖：

```bash
bash setup_comfyui.sh
```

2. 下载 FLUX.1-dev 模型：

```bash
bash deploy_comfyui.sh
```

3. 启动 ComfyUI 服务器进行手动测试（可选）：

```bash
bash start_comfyui.sh
```

4. 运行自动化性能测试：

```bash
bash run_comfyui_tests.sh
```

## 测试参数

`test_comfyui.py` 脚本支持以下参数：

- `--prompt`: 生成图像的提示词（默认：一只可爱的小猫咪在草地上玩耍）
- `--negative_prompt`: 负面提示词（默认：模糊的, 低质量的）
- `--steps`: 推理步数（默认：20）
- `--height`: 图像高度（默认：512）
- `--width`: 图像宽度（默认：512）
- `--output_dir`: 输出目录（默认：./outputs）
- `--precision`: 模型精度，可选 full (float32) 或 half (float16)（默认：half）
- `--server_url`: ComfyUI 服务器 URL（默认：http://127.0.0.1:8188）

示例：

```bash
python test_comfyui.py --precision full --steps 30 --output_dir ./outputs/custom_test
```

## 测试结果

测试结果将保存在 `./outputs/` 目录中，包括：

- 生成的图像
- 性能指标 JSON 文件
- 比较图表 (`flux_precision_comparison.png`)
- 性能测试报告 (`flux_comfyui_performance_report.md`)

## 注意事项

- 测试需要足够的 CPU 资源和内存
- 首次运行时需要下载模型，可能需要较长时间
- 如果遇到依赖问题，可能需要手动安装特定版本的依赖
