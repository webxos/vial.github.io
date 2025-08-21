FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY public/ ./public/

ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=http://localhost:8000

EXPOSE 8000

CMD ["uvicorn", "server.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
