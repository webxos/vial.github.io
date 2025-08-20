# Vial MCP Control Server - Phase 2 Guide: Quantum Wallets and Backend Stability

## 🎯 Project Overview
Phase 2 of the Vial MCP Control Server, hosted at `vial_mcp.vercel.app`, builds on Phase 1 by integrating quantum wallets with WebXOS wallet functionality, enhancing OAuth 2.0 security, and stabilizing the FastAPI/FastMCP backend. This phase prepares for future expansions (Phase 3) with training, tools, resources, and API calls, ensuring seamless Vercel and GitHub deployment. The focus is on robustness, security, testing, and self-managing DAO features.

## 📋 Prerequisites
- Completed Phase 1 setup
- Node.js 18+, Python 3.11+, Vercel CLI, Docker Desktop, Git
- PostgreSQL, SQLite, MongoDB instances
- Updated `.env` with Phase 1 variables

## 🚀 Setup Steps

### 2.1 Repository and Structure Update
- Update the GitHub repository `vial_mcp.vercel.app` with branch protection for `main`.
- Expand the `/vial/` directory structure to include Phase 2 files:
```
vial_mcp.vercel.app/
├── /vial/
│   ├── /api/                # Vercel API routes
│   │   ├── __init__.py      # API module initialization
│   │   ├── auth.py          # OAuth 2.0 endpoints (updated)
│   │   ├── dao.py           # DAO management (updated)
│   │   ├── unified_agent.py # Multi-language agent (updated)
│   │   ├── quantum_wallet.py # Quantum wallet integration
│   │   └── tools.py         # MCP tools registration
│   ├── /database/           # Database configurations
│   │   ├── __init__.py      # Database module initialization
│   │   └── data_models.py   # Database models (updated)
│   ├── /error_logging/      # Error handling utilities
│   │   ├── __init__.py      # Error logging module
│   │   └── logger.py        # Logging configuration
│   ├── /langchain/          # LangChain configurations
│   │   ├── __init__.py      # LangChain module
│   │   └── config.py        # LangChain setup
│   ├── /monitoring/         # Monitoring utilities
│   │   ├── __init__.py      # Monitoring module
│   │   └── health.py        # Health check endpoint
│   ├── /prompts/            # Prompt management
│   │   ├── __init__.py      # Prompts module
│   │   └── prompt_manager.py # Prompt handling
│   ├── /quantum/            # Quantum network components
│   │   ├── __init__.py      # Quantum module
│   │   ├── network.py       # Quantum network logic
│   │   └── wallet.py        # Quantum wallet logic
│   ├── /resources/          # Resource management
│   │   ├── __init__.py      # Resources module
│   │   └── resource_manager.py # Resource handling
│   ├── /security/           # Security configurations
│   │   ├── __init__.py      # Security module
│   │   └── middleware.py    # Security middleware
│   ├── /tests/              # Test suites
│   │   ├── __init__.py      # Tests module
│   │   ├── test_auth.py     # Authentication tests
│   │   └── test_dao.py      # DAO tests
│   ├── __init__.py          # Main module initialization (updated)
│   ├── main.py              # FastAPI backend (updated)
│   ├── requirements.txt     # Python dependencies (updated)
│   └── Dockerfile           # Docker configuration (updated)
├── /public/                 # Static assets
│   └── index.html           # Vial MCP controller interface (updated)
├── vercel.json              # Vercel configuration (updated)
├── package.json             # Node.js dependencies (updated)
├── .env.example             # Environment variables template (updated)
└── .gitignore               # Gitignore (updated)
```

### 2.2 Redo Phase 1 Core Files
- Update `/vial/main.py` for FastMCP compliance:
  ```python
  from fastapi import FastAPI
  from fastapi.middleware.cors import CORSMiddleware
  from .api import auth, dao, unified_agent, tools
  from .quantum import network
  from .security import middleware
  from .monitoring import health

  app = FastAPI(title="Vial MCP API", version="2.9.1")

  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  app.add_middleware(middleware.SecurityMiddleware)

  app.include_router(auth.router, prefix="/api/auth")
  app.include_router(dao.router, prefix="/api/dao")
  app.include_router(unified_agent.router, prefix="/api")
  app.include_router(tools.router, prefix="/api/tools")
  app.include_router(health.router, prefix="/api/monitoring")

  @app.get("/")
  async def root():
      return {"message": "Vial MCP Controller API v2.9.1", "status": "online", "time": "07:52 AM EDT, Aug 20, 2025"}
  ```
- Update `/vial/requirements.txt`:
  ```
  fastapi==0.110.0
  uvicorn==0.29.0
  pydantic==2.7.1
  sqlalchemy==2.0.30
  psycopg2-binary==2.9.9
  langchain==0.2.0
  pymongo==4.6.2
  python-jose==3.3.0
  aiohttp==3.9.5
  ```
- Update `/public/index.html` with quantum and DAO controls:
  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vial MCP Controller</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/react@18/umd/react.development.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18/umd/react-dom.development.js"></script>
  </head>
  <body class="bg-gray-900 text-green-400 font-mono">
    <div id="root" class="container mx-auto p-4">
      <h1 class="text-2xl mb-4">Vial MCP Controller</h1>
      <div id="console" class="bg-black p-4 rounded h-64 overflow-y-auto mb-4">
        <p>System initialized - 07:52 AM EDT, Aug 20, 2025</p>
      </div>
      <div id="controls">
        <button id="auth" class="bg-green-500 hover:bg-green-700 text-white py-2 px-4 rounded">Authenticate</button>
        <button id="propose" class="bg-blue-500 hover:bg-blue-700 text-white py-2 px-4 rounded">Create Proposal</button>
        <button id="quantum" class="bg-purple-500 hover:bg-purple-700 text-white py-2 px-4 rounded">Quantum Link</button>
      </div>
      <footer class="mt-4">
        <p>Vial MCP | Online | 2025 | v2.9.1</p>
      </footer>
    </div>
    <script>
      const consoleEl = document.getElementById('console');
      const log = (msg) => { consoleEl.innerHTML += `<p>${msg}</p>`; consoleEl.scrollTop = consoleEl.scrollHeight; };

      document.getElementById('auth').addEventListener('click', async () => {
        const response = await fetch('/api/auth/token', { method: 'POST', body: JSON.stringify({ username: 'admin', password: 'admin' }) });
        log(`Authentication: ${await response.text()}`);
      });

      document.getElementById('propose').addEventListener('click', async () => {
        const response = await fetch('/api/dao/proposals/create', { method: 'POST', body: JSON.stringify({ title: 'Test Proposal', description: 'Test' }) });
        log(`Proposal: ${JSON.stringify(await response.json())}`);
      });

      document.getElementById('quantum').addEventListener('click', async () => {
        const response = await fetch('/api/tools/quantum_establish_link', { method: 'POST', body: JSON.stringify({ node_a: 'QN-001', node_b: 'QN-002' }) });
        log(`Quantum: ${JSON.stringify(await response.json())}`);
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === '/' && e.target.tagName !== 'INPUT') {
          e.preventDefault();
          log('/help - Show this help\n/auth - Authenticate\n/propose - Create proposal\n/quantum - Establish quantum link');
        }
      });
    </script>
  </body>
  </html>
  ```
- Update `vercel.json` with security and timeouts:
  ```json
  {
    "version": 2,
    "builds": [
      {
        "src": "vial/main.py",
        "use": "@vercel/python"
      }
    ],
    "routes": [
      {
        "src": "/(.*)",
        "dest": "/vial/main.py"
      }
    ],
    "env": {
      "DATABASE_URL": "@database_url",
      "NEXTAUTH_SECRET": "@nextauth_secret",
      "MONGO_URL": "@mongo_url",
      "OPENAI_API_KEY": "@openai_api_key"
    },
    "headers": [
      {
        "source": "/(.*)",
        "headers": [
          { "key": "X-Content-Type-Options", "value": "nosniff" },
          { "key": "X-Frame-Options", "value": "DENY" }
        ]
      }
    ],
    "functions": {
      "timeout": 60
    }
  }
  ```
- Update `package.json`:
  ```json
  {
    "name": "vial_mcp",
    "version": "2.9.1",
    "scripts": {
      "dev": "vercel dev"
    },
    "dependencies": {
      "@vercel/python": "^3.2.0"
    }
  }
  ```
- Update `.env.example`:
  ```
  DATABASE_URL=postgresql://user:password@localhost:5432/vialmcp
  NEXTAUTH_SECRET=your-nextauth-secret-key
  MONGO_URL=mongodb://localhost:27017
  OPENAI_API_KEY=your-api-key
  QUANTUM_NODES=QN-001,QN-002,QN-003
  ```
- Update `.gitignore`:
  ```
  __pycache__/
  *.pyc
  *.pyo
  *.pyd
  env/
  venv/
  node_modules/
  npm-debug.log
  yarn-error.log
  .vercel
  .output
  *.log
  .vscode/
  .idea/
  .DS_Store
  Thumbs.db
  dist/
  build/
  coverage/
  .env
  ```

### 2.3 Quantum Wallet and WebXOS Integration
- Create `/vial/quantum/wallet.py`:
  ```python
  import uuid
  from datetime import datetime
  from typing import Dict

  class QuantumWallet:
      def __init__(self):
          self.wallets: Dict[str, Dict] = {}
          self.quantum_nodes = os.getenv("QUANTUM_NODES", "QN-001,QN-002,QN-003").split(",")

      def create_wallet(self, address: str) -> Dict:
          wallet_id = str(uuid.uuid4())
          self.wallets[wallet_id] = {
              "wallet_id": wallet_id,
              "address": address,
              "quantum_state": {"qubits": [], "entanglement": "synced"},
              "created_at": datetime.now().isoformat(),
              "last_sync": None
          }
          return self.wallets[wallet_id]

      def link_quantum(self, wallet_id: str, node_a: str, node_b: str) -> Dict:
          if wallet_id not in self.wallets or node_a not in self.quantum_nodes or node_b not in self.quantum_nodes:
              return {"error": "Invalid wallet or nodes"}
          self.wallets[wallet_id]["quantum_state"]["entanglement"] = "linked"
          self.wallets[wallet_id]["last_sync"] = datetime.now().isoformat()
          return {"wallet_id": wallet_id, "status": "linked", "time": "07:52 AM EDT, Aug 20, 2025"}

  quantum_wallet = QuantumWallet()
  ```
- Create `/vial/quantum/network.py`:
  ```python
  import os
  import random
  import asyncio
  from typing import Dict
  from datetime import datetime

  class QuantumNetwork:
      def __init__(self):
          self.entangled_pairs: Dict[str, Dict] = {}
          self.network_nodes = os.getenv("QUANTUM_NODES", "QN-001,QN-002,QN-003").split(",")

      async def establish_quantum_link(self, node_a: str, node_b: str) -> Dict:
          if node_a not in self.network_nodes or node_b not in self.network_nodes or node_a == node_b:
              return {"error": "Invalid nodes"}
          link_id = f"{node_a}_{node_b}_{random.randint(1000, 9999)}"
          self.entangled_pairs[link_id] = {
              "node_a": node_a,
              "node_b": node_b,
              "fidelity": random.uniform(0.85, 0.99),
              "state": "entangled",
              "last_sync": datetime.now().isoformat()
          }
          return {"link_id": link_id, "status": "established", "time": "07:52 AM EDT, Aug 20, 2025"}

  quantum_network = QuantumNetwork()
  ```
- Update `/vial/api/quantum_wallet.py`:
  ```python
  from fastapi import APIRouter, Depends
  from .auth import get_current_user
  from ..quantum.wallet import quantum_wallet
  from ..quantum.network import quantum_network

  router = APIRouter()

  @router.post("/wallet/create")
  async def create_wallet(address: str, current_user: dict = Depends(get_current_user)):
      return quantum_wallet.create_wallet(address)

  @router.post("/wallet/link_quantum")
  async def link_quantum(wallet_id: str, node_a: str, node_b: str, current_user: dict = Depends(get_current_user)):
      wallet_link = quantum_wallet.link_quantum(wallet_id, node_a, node_b)
      if "error" not in wallet_link:
          await quantum_network.establish_quantum_link(node_a, node_b)
      return wallet_link
  ```

### 2.4 Enhanced Security and OAuth
- Update `/vial/api/auth.py`:
  ```python
  from fastapi import APIRouter, Depends, HTTPException
  from fastapi.security import OAuth2PasswordBearer
  import jwt
  import os
  from datetime import datetime, timedelta

  router = APIRouter()
  oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

  SECRET_KEY = os.getenv("NEXTAUTH_SECRET", "default-secret")
  ALGORITHM = "HS256"
  ACCESS_TOKEN_EXPIRE_MINUTES = 30

  async def get_current_user(token: str = Depends(oauth2_scheme)):
      try:
          payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
          if datetime.utcnow() > datetime.utcfromtimestamp(payload["exp"]):
              raise HTTPException(status_code=401, detail="Token expired")
          return {"username": payload.get("sub")}
      except jwt.PyJWTError:
          raise HTTPException(status_code=401, detail="Invalid token")

  @router.post("/token")
  async def login(username: str, password: str):
      if username != "admin" or password != "admin":
          raise HTTPException(status_code=401, detail="Invalid credentials")
      expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
      token = jwt.encode({"sub": username, "exp": expire.timestamp()}, SECRET_KEY, algorithm=ALGORITHM)
      return {"access_token": token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}

  @router.get("/me")
  async def read_users_me(current_user: dict = Depends(get_current_user)):
      return current_user
  ```
- Create `/vial/security/middleware.py`:
  ```python
  from fastapi import Request, HTTPException
  from starlette.middleware.base import BaseHTTPMiddleware

  class SecurityMiddleware(BaseHTTPMiddleware):
      async def dispatch(self, request: Request, call_next):
          if request.url.path.startswith("/api"):
              auth = request.headers.get("Authorization")
              if not auth or not auth.startswith("Bearer "):
                  raise HTTPException(status_code=401, detail="Bearer token required")
          response = await call_next(request)
          response.headers["X-Content-Type-Options"] = "nosniff"
          return response
  ```

### 2.5 DAO and Unified Agent Enhancements
- Update `/vial/api/dao.py`:
  ```python
  from fastapi import APIRouter, Depends
  from pydantic import BaseModel
  from .auth import get_current_user
  import os
  from datetime import datetime, timedelta

  router = APIRouter()
  VOTING_PERIOD = 7 * 24 * 3600  # 7 days

  class ProposalCreate(BaseModel):
      title: str
      description: str

  class DAOCore:
      def __init__(self):
          self.proposals = {}
          self.members = {}
          self.audit_log = []

      def create_proposal(self, title: str, description: str, creator: str):
          proposal_id = f"PROP-{len(self.proposals) + 1}"
          self.proposals[proposal_id] = {
              "title": title,
              "description": description,
              "creator": creator,
              "voting_start": datetime.now(),
              "voting_end": datetime.now() + timedelta(seconds=VOTING_PERIOD),
              "status": "active",
              "votes": {}
          }
          self.audit_log.append({"action": "proposal_created", "proposal_id": proposal_id, "time": datetime.now().isoformat()})
          return proposal_id

      def audit_proposal(self, proposal_id: str):
          if proposal_id in self.proposals:
              self.audit_log.append({"action": "proposal_audited", "proposal_id": proposal_id, "time": datetime.now().isoformat()})
              return {"status": "audited", "details": self.proposals[proposal_id], "audit_time": datetime.now().isoformat()}
          return {"status": "not found"}

  dao = DAOCore()

  @router.post("/proposals/create")
  async def create_proposal(proposal: ProposalCreate, current_user: dict = Depends(get_current_user)):
      proposal_id = dao.create_proposal(proposal.title, proposal.description, current_user["username"])
      return {"proposal_id": proposal_id, "message": "Proposal created", "audit": dao.audit_log[-1]}

  @router.get("/proposals/{proposal_id}/audit")
  async def audit_proposal(proposal_id: str):
      return dao.audit_proposal(proposal_id)
  ```
- Update `/vial/api/unified_agent.py`:
  ```python
  from fastapi import APIRouter, Depends
  from pydantic import BaseModel
  import json
  import sqlite3
  from sqlalchemy import create_engine
  from langchain.llms import OpenAI
  import os
  import pymongo
  from typing import Dict
  from .auth import get_current_user

  router = APIRouter()
  SECRET_KEY = os.getenv("NEXTAUTH_SECRET", "default-secret")
  DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/vialmcp")
  MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
  engine = create_engine(DATABASE_URL)
  mongo_client = pymongo.MongoClient(MONGO_URL)
  db = mongo_client["vialmcp"]
  llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key"))

  class DataRequest(BaseModel):
      type: str
      content: Dict

  def translate_data(data: Dict, target_type: str) -> Dict:
      if target_type == "json":
          return json.dumps(data)
      elif target_type == "sqlite":
          conn = sqlite3.connect("/data/vialmcp.db")
          conn.execute("INSERT INTO data (content) VALUES (?)", (json.dumps(data),))
          conn.close()
          return {"status": "saved"}
      elif target_type == "postgres":
          with engine.connect() as conn:
              conn.execute("INSERT INTO data (content) VALUES (:content)", {"content": json.dumps(data)})
              conn.commit()
          return {"status": "saved"}
      elif target_type == "typescript":
          return f"const data = {json.dumps(data)};"
      elif target_type == "langchain":
          return llm(f"Process: {json.dumps(data)}")
      return data

  @router.post("/process")
  async def process_data(request: DataRequest, current_user: dict = Depends(get_current_user)):
      return translate_data(request.content, request.type)

  @router.post("/train")
  async def train_agent(data: Dict = None, current_user: dict = Depends(get_current_user)):
      if data:
          translated = translate_data(data, "langchain")
          return {"message": "Agent trained", "output": translated, "user": current_user["username"]}
      return {"message": "No data provided"}
  ```

### 2.6 MCP Tools and Resources
- Create `/vial/api/tools.py`:
  ```python
  from fastapi import APIRouter
  from ..quantum.network import quantum_network
  from ..quantum.wallet import quantum_wallet

  router = APIRouter()

  @router.post("/quantum_establish_link")
  async def establish_quantum_link(node_a: str, node_b: str):
      return await quantum_network.establish_quantum_link(node_a, node_b)

  @router.post("/quantum_measure")
  async def measure_quantum_state(link_id: str):
      # Placeholder for future quantum measurement (requires network expansion)
      return {"link_id": link_id, "status": "measured", "time": "07:52 AM EDT, Aug 20, 2025"}
  ```
- Create `/vial/resources/resource_manager.py`:
  ```python
  from typing import Dict

  class ResourceManager:
      def __init__(self):
          self.resources: Dict[str, Dict] = {}

      def add_resource(self, resource_id: str, data: Dict):
          self.resources[resource_id] = data
          return {"resource_id": resource_id, "status": "added"}

  resource_manager = ResourceManager()
  ```

### 2.7 Testing and Monitoring
- Create `/vial/tests/test_auth.py`:
  ```python
  from fastapi.testclient import TestClient
  from ..main import app

  client = TestClient(app)

  def test_login():
      response = client.post("/api/auth/token", json={"username": "admin", "password": "admin"})
      assert response.status_code == 200
      assert "access_token" in response.json()

  def test_protected():
      response = client.get("/api/me", headers={"Authorization": "Bearer dummy_token"})
      assert response.status_code == 401
  ```
- Create `/vial/tests/test_dao.py`:
  ```python
  from fastapi.testclient import TestClient
  from ..main import app

  client = TestClient(app)

  def test_create_proposal():
      response = client.post("/api/auth/token", json={"username": "admin", "password": "admin"})
      token = response.json()["access_token"]
      response = client.post("/api/dao/proposals/create", headers={"Authorization": f"Bearer {token}"}, json={"title": "Test", "description": "Test"})
      assert response.status_code == 200
      assert "proposal_id" in response.json()
  ```
- Create `/vial/monitoring/health.py`:
  ```python
  from fastapi import APIRouter
  import psutil

  router = APIRouter()

  @router.get("/health")
  async def health_check():
      return {
          "status": "healthy" if psutil.cpu_percent() < 90 and psutil.virtual_memory().percent < 90 else "unhealthy",
          "cpu_usage": psutil.cpu_percent(),
          "memory_usage": psutil.virtual_memory().percent,
          "time": "07:52 AM EDT, Aug 20, 2025"
      }
  ```

### 2.8 Docker and Deployment Prep
- Update `Dockerfile` with health checks:
  ```
  FROM python:3.11-slim

  WORKDIR /app/vial

  COPY vial/requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  COPY vial/ .

  ENV PYTHONUNBUFFERED=1
  ENV PYTHONDONTWRITEBYTECODE=1

  EXPOSE 8000

  HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/api/monitoring/health || exit 1

  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
  ```
- Update `docker-compose.yml`:
  ```
  version: '3.8'
  services:
    app:
      build: .
      ports:
        - "8000:8000"
      volumes:
        - .:/app
      env_file:
        - .env
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:8000/api/monitoring/health"]
        interval: 30s
        timeout: 3s
        retries: 3
  ```

## ✅ Success Criteria
- Quantum wallets created and linked with quantum network
- OAuth 2.0 with token expiration and security middleware functional
- DAO proposals and audits operational
- Unified agent processes all language types (JSON, TypeScript, PostgreSQL, SQLite, LangChain)
- Tools and resources registered for MCP compliance
- Tests pass for authentication and DAO
- Health monitoring endpoint active
- Docker container runs with health checks
- Vercel deployment configured and tested

## ⏰ Timeline Estimate
- **Duration**: 2-3 weeks
- **Next Phase**: Frontend development and expansions (Phase 3 Guide)

## 📝 Next Steps
Proceed to **Phase 3 Guide** for frontend integration and advanced features.