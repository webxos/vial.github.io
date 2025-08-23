# Stage 1: Build backend
FROM python:3.11-slim AS backend-builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ ./server/
COPY vial/ ./vial/

# Stage 2: Build frontend
FROM node:18-slim AS frontend-builder
WORKDIR /app
COPY package.json .
COPY public/ ./public/
COPY pages/ ./pages/
COPY styles/ ./styles/
RUN npm install && npm run build

# Stage 3: Final image
FROM python:3.11-slim
WORKDIR /app
COPY --from=backend-builder /app /app
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=frontend-builder /app/.next ./.next
COPY --from=frontend-builder /app/node_modules ./node_modules
COPY --from=frontend-builder /app/package.json ./package.json
COPY scripts/ ./scripts/
COPY public/index.html ./public/index.html
ENV PYTHONPATH=/app
ENV PORT=3000
EXPOSE 3000 8000 9090
CMD ["sh", "-c", "uvicorn server.mcp_server:app --host 0.0.0.0 --port 8000 & npm start"]
