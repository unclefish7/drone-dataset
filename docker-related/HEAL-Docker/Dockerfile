# 基于官方 CUDA/cuDNN 开发镜像（Ubuntu 22.04），包含完整 CUDA Toolkit 和 cuDNN
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 更新软件包并安装 wget、git、Python3 和 pip
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
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
# apt-get update && apt-get install -y libgl1-mesa-glx （暂时不需要用）

# 基于镜像运行容器的命令：
# docker run --privileged --gpus all -it --name heal-new --shm-size=32G -v E:\datasets:/root/HEAL/dataset -e DISPLAY=192.168.1.88:0.0 -e LIBGL_ALWAYS_INDIRECT=1 -v /tmp/.X11-unix:/tmp/.X11-unix heal-opencood:v7

# 先切换虚拟环境
# conda activate heal

# 运行训练的命令
# python opencood/tools/train.py -y opencood/hypes_yaml/opv2v/CameraOnly/camera_attfuse.yaml
# python opencood/tools/train.py -y opencood/hypes_yaml/opv2v/LiDAROnly/lidar_attfuse.yaml