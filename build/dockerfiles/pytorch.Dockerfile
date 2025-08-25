FROM nvidia/cuda:12.1.0-base-ubuntu20.04 AS pytorch_builder
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-pytorch.txt .
RUN pip install --no-cache-dir -r requirements-pytorch.txt \
    && pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
COPY server/services/ml/ /app/ml/
