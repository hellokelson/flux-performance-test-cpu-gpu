# FLUX.1-dev 模型性能测试

本项目用于评估 black-forest-labs/FLUX.1-dev 模型在 CPU (M7i) 和 GPU (G6) 环境下的性能表现。

## 目录结构

```
.
├── README.md                    # 项目说明文档
├── cpu/                         # CPU 相关脚本
│   ├── setup.sh                 # CPU 环境初始化脚本
│   ├── deploy.sh                # CPU 模型部署脚本
│   └── test_inference.py        # CPU 推理测试脚本
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

## 性能测试结果

测试结果将包括以下指标：
- 模型加载时间
- 单张图片生成时间
- 内存使用情况
- CPU/GPU 利用率
- 批量生成性能
