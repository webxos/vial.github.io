# Vial MCP Control Server - Phase 3 Guide: Resource, Tool, and Prompt Expansion

## 🎯 Project Overview
Phase 3 of the Vial MCP Control Server, hosted at `vial_mcp.vercel.app`, expands on Phases 1 and 2 by developing resources, tools, and prompts synchronized with Git commands. This phase integrates advanced languages (JSON, SQL, PostgreSQL, PyTorch, TypeScript, LangChain, Next.js, Node.js) to support API calls, server creation, and quantum wallet management within the `index.html` single control console. The build ensures seamless operation with GitHub and Vercel hosting, preparing for Phase 4 (testing and finalization) by validating all functionalities.

## 📋 Prerequisites
- Completed Phases 1 and 2 setups
- Node.js 18+, Python 3.11+, Vercel CLI, Docker Desktop, Git
- PostgreSQL, SQLite, MongoDB instances
- Updated `.env` with Phase 2 variables
- GitHub repository `vial_mcp.vercel.app` configured

## 🚀 Setup Steps

### 3.1 Repository and Structure Update
- Enhance the GitHub repository `vial_mcp.vercel.app` with Git command synchronization.
- Expand the `/vial/` directory structure to include Phase 3 files:
```
vial_mcp.vercel.app/
├── /vial/
│   ├── /api/                # Vercel API routes
│   │   ├── __init__.py      # API module initialization
│   │   ├── auth.py          # OAuth 2.0 endpoints
│   │   ├── dao.py           # DAO management
│   │   ├── unified_agent.py # Multi-language agent
│   │   ├── quantum_wallet.py # Quantum wallet integration
│   │   ├── tools.py         # MCP tools registration
│   │   ├── server.py        # Server creation API
│   │   └── wallet_api.py    # Wallet management API
│   ├── /database/           # Database configurations
│   │   ├── __init__.py      # Database module initialization
│   │   ├── data_models.py   # Database models
│   │   └── migrations.py    # Database migrations
│   ├── /error_logging/      # Error handling utilities
│   │   ├── __init__.py      # Error logging module
│   │   └── logger.py        # Logging configuration
│   ├── /langchain/          # LangChain configurations
│   │   ├── __init__.py      # LangChain module
│   │   ├── config.py        # LangChain setup
│   │   └── agent.py         # LangChain agent logic
│   ├── /monitoring/         # Monitoring utilities
│   │   ├── __init__.py      # Monitoring module
│   │   ├── health.py        # Health check endpoint
│   │   └── metrics.py       # Performance metrics
│   ├── /prompts/            # Prompt management
│   │   ├── __init__.py      # Prompts module
│   │   ├── prompt_manager.py # Prompt handling
│   │   └── prompt_templates.py # Prompt templates
│   ├── /quantum/            # Quantum network components
│   │   ├── __init__.py      # Quantum module
│   │   ├── network.py       # Quantum network logic
│   │   ├── wallet.py        # Quantum wallet logic
│   │   └── state.py         # Quantum state management
│   ├── /resources/          # Resource management
│   │   ├── __init__.py      # Resources module
│   │   ├── resource_manager.py # Resource handling
│   │   └── resource_types.py # Resource definitions
│   ├── /security/           # Security configurations
│   │   ├── __init__.py      # Security module
│   │   ├── middleware.py    # Security middleware
│   │   └── rate_limiter.py  # Rate limiting
│   ├── /tests/              # Test suites
│   │   ├── __init__.py      # Tests module
│   │   ├── test_auth.py     # Authentication tests
│   │   ├── test_dao.py      # DAO tests
│   │   ├── test_quantum.py  # Quantum tests
│   │   └── test_tools.py    # Tools tests
│   ├── /tools/              # Tool implementations
│   │   ├── __init__.py      # Tools module
│   │   ├── git_sync.py      # Git synchronization
│   │   └── server_tools.py  # Server creation tools
│   ├── __init__.py          # Main module initialization
│   ├── main.py              # FastAPI backend
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Docker configuration
├── /public/                 # Static assets
│   └── index.html           # Vial MCP controller interface
├── vercel.json              # Vercel configuration
├── package.json             # Node.js dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Gitignore
└── README.md                # Project README
```

### 3.2 Language-Specific Implementations
#### 3.2.1 JSON
- Update `/vial/api/unified_agent.py` to enhance JSON handling:
  ```python
  # ... (previous imports and router setup)
  @router.post("/json/export")
  async def export_json(data: Dict, current_user: dict = Depends(get_current_user)):
      return {"json_data": json.dumps(data), "exported_at": "08:06 AM EDT, Aug 20, 2025"}
  ```

#### 3.2.2 SQL and PostgreSQL
- Update `/vial/database/data_models.py` with DAO and wallet models:
  ```python
  from sqlalchemy import Column, Integer, String, DateTime
  from sqlalchemy.ext.declarative import declarative_base

  Base = declarative_base()

  class Wallet(Base):
      __tablename__ = "wallets"
      id = Column(Integer, primary_key=True, index=True)
      wallet_id = Column(String(36), unique=True, index=True)
      address = Column(String(36))
      quantum_state = Column(String)
      created_at = Column(DateTime, default=datetime.now)

  class Proposal(Base):
      __tablename__ = "dao_proposals"
      id = Column(Integer, primary_key=True, index=True)
      title = Column(String(500))
      description = Column(String)
      creator = Column(String(255))
      voting_start = Column(DateTime)
      voting_end = Column(DateTime)
      status = Column(String(50))
      created_at = Column(DateTime, default=datetime.now)
  ```
- Create `/vial/database/migrations.py`:
  ```python
  from sqlalchemy import create_engine
  from .data_models import Base
  import os

  engine = create_engine(os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/vialmcp"))

  def migrate():
      Base.metadata.create_all(engine)
      return {"status": "migration_complete", "time": "08:06 AM EDT, Aug 20, 2025"}
  ```

#### 3.2.3 PyTorch
- Create `/vial/tools/server_tools.py` for PyTorch-based server creation:
  ```python
  import torch
  import torch.nn as nn
  from typing import Dict

  class ServerModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc = nn.Linear(10, 2)  # Simplified model for server state

      def forward(self, x):
          return torch.sigmoid(self.fc(x))

  def create_server_model(config: Dict) -> Dict:
      model = ServerModel()
      torch.save(model.state_dict(), f"/app/vial/models/server_{config['id']}.pt")
      return {"model_id": config["id"], "status": "created", "time": "08:06 AM EDT, Aug 20, 2025"}

  server_model = ServerModel()
  ```

#### 3.2.4 TypeScript
- Update `/vial/api/server.py` with TypeScript-compatible API:
  ```python
  from fastapi import APIRouter, Depends
  from .auth import get_current_user
  from ..tools.server_tools import create_server_model

  router = APIRouter()

  @router.post("/server/create")
  async def create_server(server_id: str, config: Dict, current_user: dict = Depends(get_current_user)):
      ts_config = f"const serverConfig = {json.dumps(config)};"
      model_result = create_server_model({"id": server_id, "config": ts_config})
      return {**model_result, "typescript_config": ts_config}
  ```

#### 3.2.5 LangChain
- Update `/vial/langchain/agent.py`:
  ```python
  from langchain.llms import OpenAI
  from langchain.prompts import PromptTemplate
  import os

  llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key"))

  class LangChainAgent:
      def __init__(self):
          self.prompt = PromptTemplate(input_variables=["input"], template="Process: {input}")

      def process(self, input_data: str) -> str:
          return llm(self.prompt.format(input=input_data))

  agent = LangChainAgent()
  ```

#### 3.2.6 Next.js and Node.js
- Enhance `/public/index.html` with Next.js-like reactivity:
  ```html
  <!-- ... (previous HTML head) -->
  <body class="bg-gray-900 text-green-400 font-mono">
    <div id="root" class="container mx-auto p-4">
      <h1 class="text-2xl mb-4">Vial MCP Controller</h1>
      <div id="console" class="bg-black p-4 rounded h-64 overflow-y-auto mb-4"></div>
      <div id="controls">
        <button id="auth" class="bg-green-500 hover:bg-green-700 text-white py-2 px-4 rounded">Authenticate</button>
        <button id="propose" class="bg-blue-500 hover:bg-blue-700 text-white py-2 px-4 rounded">Create Proposal</button>
        <button id="quantum" class="bg-purple-500 hover:bg-purple-700 text-white py-2 px-4 rounded">Quantum Link</button>
        <button id="server" class="bg-yellow-500 hover:bg-yellow-700 text-white py-2 px-4 rounded">Create Server</button>
        <button id="wallet" class="bg-red-500 hover:bg-red-700 text-white py-2 px-4 rounded">Create Wallet</button>
      </div>
      <footer class="mt-4">
        <p>Vial MCP | Online | 2025 | v2.9.2</p>
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
        const response = await fetch('/api/dao/proposals/create', { method: 'POST', body: JSON.stringify({ title: 'Test', description: 'Test' }) });
        log(`Proposal: ${JSON.stringify(await response.json())}`);
      });

      document.getElementById('quantum').addEventListener('click', async () => {
        const response = await fetch('/api/tools/quantum_establish_link', { method: 'POST', body: JSON.stringify({ node_a: 'QN-001', node_b: 'QN-002' }) });
        log(`Quantum: ${JSON.stringify(await response.json())}`);
      });

      document.getElementById('server').addEventListener('click', async () => {
        const response = await fetch('/api/server/create', { method: 'POST', body: JSON.stringify({ id: 'SVR-001', config: { nodes: 2 } }) });
        log(`Server: ${JSON.stringify(await response.json())}`);
      });

      document.getElementById('wallet').addEventListener('click', async () => {
        const response = await fetch('/api/wallet/create', { method: 'POST', body: JSON.stringify({ address: 'e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d' }) });
        log(`Wallet: ${JSON.stringify(await response.json())}`);
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === '/' && e.target.tagName !== 'INPUT') {
          e.preventDefault();
          log('/help - Show this help\n/auth - Authenticate\n/propose - Create proposal\n/quantum - Establish quantum link\n/server - Create server\n/wallet - Create wallet');
        }
      });
    </script>
  </body>
  </html>
  ```

### 3.3 Git Command Synchronization
- Create `/vial/tools/git_sync.py`:
  ```python
  import subprocess
  import os
  from typing import Dict

  class GitSync:
      def __init__(self, repo_path: str = "."):
          self.repo_path = repo_path

      def sync(self) -> Dict:
          try:
              subprocess.run(["git", "pull"], cwd=self.repo_path, check=True)
              subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
              subprocess.run(["git", "commit", "-m", f"Sync at {datetime.now().isoformat()}"], cwd=self.repo_path, check=True)
              subprocess.run(["git", "push"], cwd=self.repo_path, check=True)
              return {"status": "synced", "time": "08:06 AM EDT, Aug 20, 2025"}
          except subprocess.CalledProcessError as e:
              return {"status": "error", "message": str(e), "time": "08:06 AM EDT, Aug 20, 2025"}

  git_sync = GitSync()
  ```

### 3.4 Resource, Tool, and Prompt Development
- Update `/vial/resources/resource_manager.py` with advanced types:
  ```python
  from typing import Dict
  import json

  class ResourceManager:
      def __init__(self):
          self.resources: Dict[str, Dict] = {}

      def add_resource(self, resource_id: str, data: Dict, resource_type: str = "default"):
          self.resources[resource_id] = {"type": resource_type, "data": data, "created_at": datetime.now().isoformat()}
          return {"resource_id": resource_id, "status": "added", "type": resource_type}

      def get_resource(self, resource_id: str) -> Dict:
          return self.resources.get(resource_id, {"status": "not_found"})

  resource_manager = ResourceManager()
  ```
- Update `/vial/api/tools.py` with resource integration:
  ```python
  from fastapi import APIRouter
  from ..quantum.network import quantum_network
  from ..quantum.wallet import quantum_wallet
  from ..resources.resource_manager import resource_manager
  from ..tools.git_sync import git_sync

  router = APIRouter()

  @router.post("/quantum_establish_link")
  async def establish_quantum_link(node_a: str, node_b: str):
      result = await quantum_network.establish_quantum_link(node_a, node_b)
      resource_manager.add_resource(f"link_{result['link_id']}", result, "quantum")
      return result

  @router.post("/quantum_measure")
  async def measure_quantum_state(link_id: str):
      result = {"link_id": link_id, "status": "measured", "time": "08:06 AM EDT, Aug 20, 2025"}
      resource_manager.add_resource(f"measure_{link_id}", result, "quantum")
      return result

  @router.post("/git/sync")
  async def git_sync_endpoint():
      return git_sync.sync()
  ```
- Update `/vial/prompts/prompt_manager.py`:
  ```python
  from typing import Dict
  from langchain.prompts import PromptTemplate

  class PromptManager:
      def __init__(self):
          self.prompts: Dict[str, PromptTemplate] = {}

      def add_prompt(self, prompt_id: str, template: str, input_variables: list):
          self.prompts[prompt_id] = PromptTemplate(input_variables=input_variables, template=template)
          return {"prompt_id": prompt_id, "status": "added"}

      def get_prompt(self, prompt_id: str) -> Dict:
          prompt = self.prompts.get(prompt_id)
          return {"prompt": prompt.template if prompt else None, "status": "retrieved"} if prompt_id in self.prompts else {"status": "not_found"}

  prompt_manager = PromptManager()
  ```
- Update `/vial/prompts/prompt_templates.py`:
  ```python
  from .prompt_manager import prompt_manager

  prompt_manager.add_prompt("dao_proposal", "Create a DAO proposal with title: {title} and description: {description}", ["title", "description"])
  prompt_manager.add_prompt("quantum_link", "Establish a quantum link between {node_a} and {node_b}", ["node_a", "node_b"])
  ```

### 3.5 Wallet and Server Creation
- Create `/vial/api/wallet_api.py`:
  ```python
  from fastapi import APIRouter, Depends
  from .auth import get_current_user
  from ..quantum.wallet import quantum_wallet
  from ..tools.server_tools import create_server_model

  router = APIRouter()

  @router.post("/wallet/create")
  async def create_wallet(address: str, current_user: dict = Depends(get_current_user)):
      wallet = quantum_wallet.create_wallet(address)
      return wallet

  @router.post("/wallet/export")
  async def export_wallet(wallet_id: str, current_user: dict = Depends(get_current_user)):
      if wallet_id in quantum_wallet.wallets:
          return {"wallet": quantum_wallet.wallets[wallet_id], "exported_at": "08:06 AM EDT, Aug 20, 2025"}
      return {"status": "not_found"}
  ```
- Update `/vial/api/server.py` with wallet integration:
  ```python
  from fastapi import APIRouter, Depends
  from .auth import get_current_user
  from ..tools.server_tools import create_server_model
  from ..quantum.wallet import quantum_wallet

  router = APIRouter()

  @router.post("/server/create")
  async def create_server(server_id: str, config: Dict, wallet_id: str, current_user: dict = Depends(get_current_user)):
      if wallet_id not in quantum_wallet.wallets:
          return {"error": "Wallet not found"}
      ts_config = f"const serverConfig = {json.dumps(config)};"
      model_result = create_server_model({"id": server_id, "config": ts_config})
      quantum_wallet.wallets[wallet_id]["last_sync"] = datetime.now().isoformat()
      return {**model_result, "typescript_config": ts_config, "wallet_sync": wallet_id}
  ```

### 3.6 Testing and Validation
- Update `/vial/tests/test_quantum.py`:
  ```python
  from fastapi.testclient import TestClient
  from ..main import app

  client = TestClient(app)

  def test_quantum_link():
      response = client.post("/api/tools/quantum_establish_link", json={"node_a": "QN-001", "node_b": "QN-002"})
      assert response.status_code == 200
      assert "link_id" in response.json()

  def test_wallet_create():
      response = client.post("/api/auth/token", json={"username": "admin", "password": "admin"})
      token = response.json()["access_token"]
      response = client.post("/api/wallet/create", headers={"Authorization": f"Bearer {token}"}, json={"address": "e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d"})
      assert response.status_code == 200
      assert "wallet_id" in response.json()
  ```
- Update `/vial/tests/test_tools.py`:
  ```python
  from fastapi.testclient import TestClient
  from ..main import app

  client = TestClient(app)

  def test_git_sync():
      response = client.post("/api/tools/git/sync")
      assert response.status_code == 200
      assert "status" in response.json()

  def test_server_create():
      response = client.post("/api/auth/token", json={"username": "admin", "password": "admin"})
      token = response.json()["access_token"]
      response = client.post("/api/server/create", headers={"Authorization": f"Bearer {token}"}, json={"server_id": "SVR-001", "config": {"nodes": 2}, "wallet_id": "dummy"})
      assert response.status_code == 200
      assert "model_id" in response.json()
  ```

### 3.7 Docker and Vercel Optimization
- Update `Dockerfile` with resource limits:
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
- Update `vercel.json` with advanced routing:
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
        "src": "/api/(.*)",
        "dest": "/vial/main.py"
      },
      {
        "src": "/(.*)",
        "dest": "/public/index.html"
      }
    ],
    "env": {
      "DATABASE_URL": "@database_url",
      "NEXTAUTH_SECRET": "@nextauth_secret",
      "MONGO_URL": "@mongo_url",
      "OPENAI_API_KEY": "@openai_api_key",
      "QUANTUM_NODES": "@quantum_nodes"
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

### 3.8 Documentation and README Update
- Update `vial_mcp.vercel.app/vial/README.md`:
  ```markdown
  # Vial MCP

  Vial MCP is a secure, lightweight FastAPI backend with Next.js frontend, deployed at `vial_mcp.vercel.app` on Vercel. It features a self-managing DAO, quantum wallets, and multi-language support (JSON, SQL, PostgreSQL, PyTorch, TypeScript, LangChain, Next.js, Node.js).

  ## Features
  - **MCP Compliance**: JSON-RPC 2.0, tools, resources, prompts
  - **DAO Integration**: Self-managing proposals and audits
  - **Quantum Wallets**: Seamless integration with quantum network
  - **Unified Agent**: Handles JSON, TypeScript, PostgreSQL, SQLite, LangChain
  - **Server Creation**: PyTorch-based server models
  - **Git Sync**: Automated repository synchronization
  - **Dockerized**: Scalable deployment on Vercel

  ## Prerequisites
  - Python 3.11+
  - Node.js 18+
  - Docker
  - Vercel CLI
  - PostgreSQL, SQLite, MongoDB

  ## Setup
  1. Clone the repository:
     ```bash
     git clone https://github.com/your-org/vial_mcp.vercel.app.git
     cd vial_mcp.vercel.app
     ```
  2. Install dependencies:
     ```bash
     pip install -r vial/requirements.txt
     npm install
     ```
  3. Configure `.env` from `.env.example`.
  4. Run locally:
     ```bash
     docker-compose up --build
     ```
  5. Deploy to Vercel:
     ```bash
     vercel --prod
     ```

  ## Usage
  - Access via `/public/index.html`
  - Use `/help` for commands
  - Manage wallets, servers, and proposals via API calls

  ## Project Structure
  ```
  vial_mcp.vercel.app/
  ├── /vial/ (all Python and API files)
  ├── /public/ (static assets)
  ├── vercel.json
  ├── package.json
  ├── .env.example
  ├── .gitignore
  └── README.md
  ```

  ## Contributing
  1. Fork the repo
  2. Create a branch (`git checkout -b feature/your-feature`)
  3. Commit changes (`git commit -m 'Add feature'`)
  4. Push (`git push origin feature/your-feature`)
  5. Open a PR

  ## License
  MIT License
  ```

## ✅ Success Criteria
- Resources, tools, and prompts synchronized with Git commands
- API calls functional for all languages (JSON, SQL, PostgreSQL, PyTorch, TypeScript, LangChain, Next.js, Node.js)
- Servers and wallets created seamlessly via `index.html`
- GitHub and Vercel hosting fully integrated
- All Phase 1 and 2 functionalities preserved and enhanced
- Tests pass for quantum, tools, and wallet/server creation
- Documentation updated with new features

## ⏰ Timeline Estimate
- **Duration**: 3-4 weeks
- **Next Phase**: Testing and finalization (Phase 4 Guide)

## 📝 Next Steps
Proceed to **Phase 4 Guide** for comprehensive testing and deployment.