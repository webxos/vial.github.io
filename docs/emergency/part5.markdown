# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 5 (Deployment)

**Objective**: Deploy the WebXOS backend using Docker and Helm, with Prometheus monitoring.

**Instructions for LLM**:
1. Create Dockerfiles for base and PyTorch environments.
2. Set up Helm deployment.
3. Configure Prometheus for monitoring.
4. Integrate with `server/main.py`.

## Step 1: Create Deployment Files

### build/dockerfiles/mcp-final.Dockerfile
```dockerfile
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
```

### build/dockerfiles/pytorch.Dockerfile
```dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu20.04 AS pytorch_builder
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-pytorch.txt .
RUN pip install --no-cache-dir -r requirements-pytorch.txt \
    && pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
COPY server/services/ml/ /app/ml/
```

### .github/workflows/deploy.yml
```yaml
name: Deploy
on:
  push:
    branches: [main]
permissions:
  contents: read
  issues: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-mcp.txt -r requirements-pytorch.txt
      - name: Set up Helm
        uses: azure/setup-helm@v4
        with:
          version: '3.14.0'
      - name: Update Helm dependencies
        run: helm dependency update ./helm/webxos
      - name: Deploy to Kubernetes
        env:
          KUBE_CONFIG: ${{ secrets.KUBE_CONFIG }}
        run: |
          echo "$KUBE_CONFIG" | base64 -d > kubeconfig
          helm upgrade --install webxos ./helm/webxos --kubeconfig kubeconfig
      - name: Upload deployment artifacts
        uses: actions/upload-artifact@v4
        with:
          name: deployment-artifacts
          path: ./helm/webxos/
```

### prometheus.yml
```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'mcp-server'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'mcp-api'
  - job_name: 'spacex-api-metrics'
    static_configs:
      - targets: ['api.spacexdata.com:443']
        labels:
          service: 'spacex-api'
  - job_name: 'nasa-api-metrics'
    static_configs:
      - targets: ['api.nasa.gov:443']
        labels:
          service: 'nasa-api'
```

### server/main.py
```python
from fastapi import FastAPI
from server.api.auth_endpoint import router as auth_router
from server.services.nasa_service import router as nasa_router
from server.services.spacex_service import router as spacex_router
from server.services.github_service import router as github_router
from server.services.langchain_service import router as langchain_router
from server.webxos_wallet import WebXOSWallet
from fastapi import Depends, HTTPException
from server.api.auth_endpoint import verify_token
from pydantic import BaseModel

app = FastAPI(title="WebXOS 2025 Vial MCP SDK", version="1.2.0")
app.include_router(auth_router)
app.include_router(nasa_router)
app.include_router(spacex_router)
app.include_router(github_router)
app.include_router(langchain_router)

# Wallet instance
wallet_manager = WebXOSWallet(password="secure_wallet_password")

class WalletCreate(BaseModel):
    address: str
    private_key: str
    balance: float = 0.0

@app.post("/mcp/wallet/create", tags=["wallet"])
async def create_wallet(wallet_data: WalletCreate, token: dict = Depends(verify_token)):
    try:
        wallet = wallet_manager.create_wallet(
            address=wallet_manager.sanitize_input(wallet_data.address),
            private_key=wallet_manager.sanitize_input(wallet_data.private_key),
            balance=wallet_data.balance
        )
        return {"address": wallet.address, "balance": wallet.balance}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/mcp/wallet/{address}", tags=["wallet"])
async def get_wallet(address: str, token: dict = Depends(verify_token)):
    address = wallet_manager.sanitize_input(address)
    if address not in wallet_manager.wallets:
        raise HTTPException(status_code=404, detail="Wallet not found")
    wallet = wallet_manager.wallets[address]
    return {"address": wallet.address, "balance": wallet.balance}
```

## Step 2: Deploy and Monitor
```bash
docker build -f build/dockerfiles/mcp-final.Dockerfile -t webxos-mcp .
docker build -f build/dockerfiles/pytorch.Dockerfile -t webxos-pytorch .
docker run -p 8000:8000 --env-file .env webxos-mcp
docker run -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
helm install webxos ./helm/webxos
```

## Validation
```bash
curl http://localhost:8000/mcp/auth/login
curl http://localhost:9090/api/v1/query?query=up
```

**Completion**: Backend rebuild complete.