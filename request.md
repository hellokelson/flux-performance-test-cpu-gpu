我是一名技术人员，我需要使用ComfyUI评估 black-forest-labs/FLUX.1-dev 模型在 CPU和GPU下的性能表现。需要分别测试20步、5步下的性能情况。

CPU 模式下 启用amx加速器。
GPU 模式下，使用GPU显卡。

## 环境准备
请帮我启动2台EC2：
1. CPU是EC2的M7i.8xlarge机型。
2. GPU是EC的G6.4xlarge机型。
3. ARM GPU是EC的R8G.8xlarge机型。

Region: us-west-2
VPC ID：vpc-04a235b9b191b851f
SUBNET ID；subnet-001216118e2e41501
PEM FIle Path: /Users/zhangkap/aws/doc/authentication/ssh/zk-us-west-2.pem
Access Key Name: zk-us-west-2
安全组：sg-0675225b689027fa6
EBS: GP3, 600GB
AMI InFo
- CPU 
    AMI ID：ami-05572e392e80aee89
    AMI name： al2023-ami-2023.7.20250414.0-kernel-6.1-x86_64
- GPU 
    - AMI ID：ami-05572e392e80aee89
    - AMI Name： al2023-ami-2023.7.20250414.0-kernel-6.1-x86_64

## 部署ComfyUI
GPU的EC2需要 安装nvidia driver、cuda，参考：https://repost.aws/articles/ARwfQMxiC-QMOgWykD9mco1w/how-do-i-install-nvidia-gpu-driver-cuda-toolkit-nvidia-container-toolkit-on-amazon-ec2-instances-running-amazon-linux-2023-al2023

huggingface model access token: xxxxxx

## 执行测试
我需要你帮我分别写一些脚本，在EC2上初始化环境、部署FLUX.1-dev模型。
在前面启动的2台机器上，根据CPU、GPU的EC2和对应的测试脚本，分别执行。

## 分析结果
汇总结果并分析。