FROM python:3.11-slim AS base

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install octokit.py emscripten

COPY server/ .
COPY public/ ./public
COPY web/ ./web

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    openjdk-11-jdk \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
