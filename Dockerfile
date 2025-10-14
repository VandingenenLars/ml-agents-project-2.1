FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl wget \
    build-essential cmake ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip setuptools wheel


RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


RUN pip install mlagents mlagents-envs tensorboard matplotlib pandas seaborn scikit-learn

WORKDIR /workspace

CMD ["/bin/bash"]
