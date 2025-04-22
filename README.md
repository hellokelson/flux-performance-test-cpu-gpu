# FLUX.1-dev 模型性能测试

本项目用于评估 black-forest-labs/FLUX.1-dev 模型在 CPU 和 GPU 环境下的性能表现。测试分别在 20 步和 5 步的情况下进行，以比较不同步数对性能的影响。

## 测试环境

- **CPU 环境**: Amazon EC2 M7i.8xlarge 实例，启用 AMX 加速器
- **GPU 环境**: Amazon EC2 G6.4xlarge 实例，使用 NVIDIA GPU

## 项目结构

```
.
├── README.md                   # 项目说明文档
├── launch_instances.sh         # 启动 EC2 实例的脚本
├── cpu_setup.sh                # CPU 实例环境配置和测试脚本
├── gpu_setup.sh                # GPU 实例环境配置和测试脚本
├── analyze_results.py          # 结果分析脚本
├── cleanup.sh                  # 资源清理脚本
└── run_all.sh                  # 一键执行所有步骤的脚本
```

## 使用方法

### 一键执行所有步骤

```bash
./run_all.sh
```

这个脚本会依次执行以下操作：
1. 启动 CPU 和 GPU 实例
2. 配置 CPU 实例环境并执行测试
3. 配置 GPU 实例环境并执行测试
4. 分析测试结果
5. 询问是否清理资源（终止 EC2 实例）

### 单独执行各步骤

1. 启动实例：
```bash
./launch_instances.sh
```

2. 配置 CPU 实例并执行测试：
```bash
./cpu_setup.sh
```

3. 配置 GPU 实例并执行测试：
```bash
./gpu_setup.sh
```

4. 分析结果：
```bash
python3 analyze_results.py
```

5. 清理资源：
```bash
./cleanup.sh
```

## 测试内容

测试使用 ComfyUI 框架，通过 API 调用执行 FLUX.1-dev 模型的图像生成任务。测试内容包括：

- 在 CPU 和 GPU 环境下分别执行 20 步和 5 步的图像生成
- 记录总执行时间和每步平均时间
- 计算 GPU 相对于 CPU 的加速比

## 结果输出

测试完成后，会生成以下输出：

- `cpu_test_results.json`: CPU 测试结果
- `gpu_test_results.json`: GPU 测试结果
- `performance_summary.json`: 性能测试总结
- `flux_performance_comparison.png`: 性能对比图表

## 注意事项

- 请确保您有足够的 AWS 权限来创建和管理 EC2 实例
- 测试完成后，请记得终止 EC2 实例以避免不必要的费用
- GPU 实例的配置可能需要较长时间，请耐心等待
