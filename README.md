# FLUX.1-dev 模型性能测试

本项目用于评估 black-forest-labs/FLUX.1-dev 模型在 CPU (M7i) 和 GPU (G6) 环境下的性能表现。

## 目录结构

```
.
├── README.md                    # 项目说明文档
├── cpu/                         # CPU 相关脚本
│   ├── setup.sh                 # CPU 环境初始化脚本
│   ├── deploy.sh                # CPU 模型部署脚本
│   ├── test_inference.py        # CPU 推理测试脚本
│   └── run_amx_tests.sh         # Intel AMX 加速器测试批处理脚本
├── gpu/                         # GPU 相关脚本
│   ├── setup.sh                 # GPU 环境初始化脚本
│   ├── deploy.sh                # GPU 模型部署脚本
│   └── test_inference.py        # GPU 推理测试脚本
└── common/                      # 通用脚本
    ├── utils.py                 # 通用工具函数
    └── benchmark.py             # 性能基准测试脚本
```

## 使用方法

### CPU 环境 (M7i)

#### 基本流程

1. 初始化环境：
```bash
cd cpu
bash setup.sh
```

2. 部署模型：
```bash
bash deploy.sh
```

3. 运行推理测试：
```bash
python test_inference.py
```

#### 测试选项

`test_inference.py` 脚本支持多种命令行参数，可以自定义测试配置：

```bash
python test_inference.py --help
```

常用参数：
- `--precision`: 选择模型精度，可选值为 "float32"、"float16" 或 "bfloat16"
- `--quantize`: 启用模型量化（INT8）
- `--batch_size`: 设置批处理大小
- `--num_inference_steps`: 设置推理步数
- `--prompt`: 自定义生成图像的提示词
- `--output_dir`: 指定输出目录

示例：
```bash
# 使用 float16 精度测试
python test_inference.py --precision float16 --output_dir ./outputs/float16

# 使用 bfloat16 精度测试（适用于支持 Intel AMX 的 CPU）
python test_inference.py --precision bfloat16 --output_dir ./outputs/bfloat16

# 使用量化模型测试
python test_inference.py --precision float32 --quantize --output_dir ./outputs/int8
```

#### Intel AMX 加速器测试

对于支持 Intel AMX 加速器的 CPU（如 M7i），可以使用 `run_amx_tests.sh` 脚本自动测试不同精度并生成比较报告：

```bash
bash run_amx_tests.sh
```

此脚本会：
1. 自动测试 float32、float16、bfloat16 和 INT8 量化模型
2. 生成性能比较图表
3. 创建详细的性能测试报告

测试结果将保存在 `./outputs/` 目录下，包括：
- 生成的图像
- 性能指标 JSON 文件
- 比较图表 (`precision_comparison.png`)
- 性能测试报告 (`amx_performance_report.md`)

### GPU 环境 (G6)

1. 初始化环境：
```bash
cd gpu
bash setup.sh
```

2. 部署模型：
```bash
bash deploy.sh
```

3. 运行推理测试：
```bash
python test_inference.py
```

GPU 环境的 `test_inference.py` 脚本也支持与 CPU 版本相同的命令行参数。

## 性能测试结果

测试结果将包括以下指标：
- 模型加载时间
- 单张图片生成时间
- 内存使用情况
- CPU/GPU 利用率
- 批量生成性能

## 环境要求

### CPU 环境
- Amazon Linux 2023
- Python 3.9+
- PyTorch 2.0+
- 推荐: Intel 第四代至强处理器 (M7i) 支持 AMX 指令集

### GPU 环境
- Amazon Linux 2023
- Python 3.9+
- PyTorch 2.0+ with CUDA 11.8
- NVIDIA 驱动
- 推荐: AWS G6 实例

## 故障排除

如果遇到依赖项问题，可以尝试手动安装特定版本的依赖：

```bash
pip install protobuf==3.20.3 sentencepiece==0.1.99 tokenizers==0.13.3
pip install huggingface_hub==0.16.4
pip install diffusers==0.21.4 transformers==4.30.2
```

如果模型下载失败，可以尝试使用 Hugging Face CLI 下载：

```bash
pip install huggingface_hub
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ./models/FLUX.1-dev
```
