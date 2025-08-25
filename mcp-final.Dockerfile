FROM python:3.11-slim AS base
RUN apt-get update && apt-get install -y \
    build-essential cmake g++ sqlite3 libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-mcp.txt .
RUN pip install --no-cache-dir -r requirements-mcp.txt

FROM base AS final
COPY server/ /app/server/
COPY public/ /app/public/
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
