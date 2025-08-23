FROM python:3.11-slim

WORKDIR /app

# Install Node.js
RUN apt-get update && apt-get install -y nodejs npm

# Copy Node.js dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server/ ./server/
COPY public/ ./public/

# Security: Run as non-root user
RUN useradd -m vial
USER vial

# Expose ports
EXPOSE 8000 4455

# Start FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
