FROM python:3.11-slim AS base

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
    emscripten \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
