FROM python:3.11-slim AS backend-builder
WORKDIR /app
COPY requirements-mcp.txt .
COPY requirements-langchain-v1.txt .
COPY requirements-langchain-v2.txt .
COPY requirements-pytorch.txt .
RUN pip install --no-cache-dir -r requirements-mcp.txt \
    && pip install --no-cache-dir -r requirements-langchain-v1.txt \
    && pip install --no-cache-dir -r requirements-langchain-v2.txt \
    && pip install --no-cache-dir -r requirements-pytorch.txt --index-url https://download.pytorch.org/whl/cu121
COPY server/ ./server/
COPY vial.html .
COPY vialfolder/ ./vialfolder/

FROM node:18-alpine AS frontend-builder
WORKDIR /app
COPY package.json .
RUN npm install --legacy-peer-deps
COPY vial.html .
COPY vialfolder/ ./vialfolder/
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY --from=backend-builder /app/server/ ./server/
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=frontend-builder /app/dist/ ./dist/
EXPOSE 8000 3000
ENV VIAL_API_URL=http://localhost:8000
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port 8000 & npx serve dist -p 3000"]
