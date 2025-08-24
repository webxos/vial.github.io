```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY package.json package-lock.json ./
RUN apt-get update && apt-get install -y nodejs npm
COPY . .
RUN npm ci && npm run build
RUN pip install modelcontextprotocol qiskit qiskit-aer sqlalchemy alembic httpx torch psycopg2-binary
RUN alembic upgrade head
EXPOSE 8000
CMD ["python", "server/mcp_server.py"]
```
