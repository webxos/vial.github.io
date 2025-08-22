# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install frontend dependencies
COPY public/ ./public/
RUN npm install --prefix public/ three

# Copy application code
COPY server/ ./server/
COPY scripts/ ./scripts/
COPY public/js/threejs_integrations.js ./public/js/threejs_integrations.js

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SQLALCHEMY_DATABASE_URL=sqlite:///vial.db
ENV REDIS_HOST=redis
ENV WEBXOS_WALLET_ADDRESS=e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d

# Expose ports
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "server.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
