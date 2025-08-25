# Stage 1: Build WASM modules
FROM node:18 AS wasm-builder
WORKDIR /app
COPY server/wasm/package.json server/wasm/package-lock.json ./server/wasm/
RUN cd server/wasm && npm install && npm run build

# Stage 2: Build Python app
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=wasm-builder /app/server/wasm/pkg ./server/wasm/pkg
EXPOSE 3000 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "3000"]
