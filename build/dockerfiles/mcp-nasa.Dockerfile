# Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-pytorch.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir -r requirements-pytorch.txt --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install earthaccess pangeo-notebook requests tenacity

# Copy application code
COPY . .

# Install Node.js dependencies
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install --legacy-peer-deps

# Configure environment
ENV PYTHONUNBUFFERED=1
ENV VIAL_API_URL=http://localhost:8000

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/ || exit 1

# Run application
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port 8000 & npm run start"]
