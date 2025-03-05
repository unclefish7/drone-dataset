# 基于官方 CUDA/cuDNN 开发镜像（Ubuntu 22.04），包含完整 CUDA Toolkit 和 cuDNN
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 更新软件包并安装 wget、git、Python3 和 pip
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

# 克隆 HEAL 仓库到 /root
RUN git clone https://github.com/yifanlu0227/HEAL.git /root/HEAL

# 设置工作目录
WORKDIR /root
# 先按照仓库里的教程一步步把前置装好（包括新开一个虚拟环境），再安装其他依赖
# apt-get update && apt-get install -y libgl1-mesa-glx
#
# 最后运行的命令：
# docker run --privileged --gpus all -it --name heal --shm-size=32G -v "D:\datasets":/root/HEAL/dataset heal-custom:v3
#