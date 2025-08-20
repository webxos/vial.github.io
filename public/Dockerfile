FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    sqlite3 \
    libsqlite3-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt] pymongo redis sqlalchemy qiskit octokit.py emscripten

# Copy application code
COPY server/ .
COPY public/ ./public

# Initialize database
RUN python -c "from server.services.database import init_db; init_db()"

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "server.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
