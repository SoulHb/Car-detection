# Use Ubuntu as the base image
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

RUN apt-get update

# Install Python 3.10
RUN apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip

# Install PyTorch 1.13
RUN pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116


# Copy  and install other requirements
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY . .
RUN pip3 install -r ./requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6  -y