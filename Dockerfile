# Base image with Python and Node.js for full stack
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY mcp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Build frontend assets
WORKDIR /app/public
RUN npm install && npm run build

# Final image
FROM python:3.11-slim

WORKDIR /app

# Copy built assets and Python files
COPY --from=builder /app /app

# Install runtime dependencies
RUN pip install --no-cache-dir -r mcp/requirements.txt

# Configure environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DATABASE_URL="sqlite:///./vialmcp.db"

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/api/monitoring/health || exit 1

# Run application
CMD ["uvicorn", "mcp.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
