# Vial MCP Control Server - Phase 1 Guide: Project Setup & Repository Configuration

## 🎯 Project Overview
The Vial MCP Control Server, hosted at `vial_mcp.vercel.app`, is a self-managing DAO-integrated system compliant with Anthropic's Model Context Protocol (MCP). This phase sets up the repository, directory structure, Docker configuration, and initial files to prepare for backend development. The system excludes financial features, focusing on DAO governance, wallet management, and multi-language support (JSON, TypeScript, PostgreSQL, SQLite, LangChain) with Next.js and Node.js.

## 📋 Prerequisites
- **Tools**: Node.js 18+, Python 3.11+, Vercel CLI (`npm i -g vercel`), Docker Desktop, Git
- **Repository**: New GitHub repository (`vial_mcp.vercel.app`)
- **Environment**: Vercel account, PostgreSQL, SQLite, MongoDB

## 🚀 Setup Steps

### 1.1 Repository Initialization
- Create a new GitHub repository named `vial_mcp.vercel.app`.
- Initialize with a `README.md`, `.gitignore`, and branch protection for `main`.
- Configure repository secrets for deployment keys.
  ```bash
  git init
  git add .
  git commit -m "Initial commit"
  git remote add origin https://github.com/your-org/vial_mcp.vercel.app.git
  git push -u origin main
  ```

### 1.2 Directory Structure Creation
Set up the following structure under the `/vial/` folder:
```
vial_mcp.vercel.app/
├── /vial/
│   ├── /api/                # Vercel API routes
│   │   ├── __init__.py      # API module initialization
│   │   ├── auth.py          # OAuth 2.0 endpoints
│   │   └── dao.py           # DAO management
│   ├── /database/           # Database configurations
│   │   ├── __init__.py      # Database module initialization
│   │   └── data_models.py   # Database models
│   ├── /error_logging/      # Error handling utilities
│   ├── /langchain/          # LangChain configurations
│   ├── /monitoring/         # Monitoring utilities
│   ├── /prompts/            # Prompt management
│   ├── /quantum/            # Quantum network components
│   ├── /resources/          # Resource management
│   ├── /security/           # Security configurations
│   ├── /tests/              # Test suites
│   ├── __init__.py          # Main module initialization
│   ├── main.py              # FastAPI backend
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Docker configuration
├── /public/                 # Static assets
│   └── index.html           # Vial MCP controller interface
├── vercel.json              # Vercel configuration
├── package.json             # Node.js dependencies
└── .env.example             # Environment variables template
```

### 1.3 Environment Configuration
- Create `.env.example` with:
  ```
  DATABASE_URL=postgresql://user:password@localhost:5432/vialmcp
  NEXTAUTH_SECRET=your-nextauth-secret-key
  MONGO_URL=mongodb://localhost:27017
  OPENAI_API_KEY=your-api-key
  ```
- Copy to `.env` locally and configure variables.

### 1.4 Docker Configuration
- Create `Dockerfile` to containerize the backend:
  ```
  FROM python:3.11-slim

  WORKDIR /app/vial

  COPY vial/requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  COPY vial/ .

  ENV PYTHONUNBUFFERED=1
  ENV PYTHONDONTWRITEBYTECODE=1

  EXPOSE 8000

  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
  ```
- Add `docker-compose.yml` for local development:
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
  ```

### 1.5 Python Module Initialization
- Create `/vial/__init__.py`:
  ```python
  from .main import app
  from .api import auth, dao, unified_agent

  __all__ = ["app", "auth", "dao", "unified_agent"]
  ```
- Create `/vial/api/__init__.py`:
  ```python
  from .auth import router as auth_router
  from .dao import router as dao_router
  from .unified_agent import router as unified_agent_router

  __all__ = ["auth_router", "dao_router", "unified_agent_router"]
  ```
- Create `/vial/database/__init__.py`:
  ```python
  from .data_models import Base

  __all__ = ["Base"]
  ```

### 1.6 Initial Files
- Create `/vial/main.py`:
  ```python
  from fastapi import FastAPI
  from fastapi.middleware.cors import CORSMiddleware

  app = FastAPI(title="Vial MCP API", version="2.9.0")

  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  @app.get("/")
  async def root():
      return {"message": "Vial MCP Controller API v2.9.0", "status": "online", "time": "07:46 AM EDT, Aug 20, 2025"}
  ```
- Create `/vial/requirements.txt`:
  ```
  fastapi==0.110.0
  uvicorn==0.29.0
  pydantic==2.7.1
  sqlalchemy==2.0.30
  psycopg2-binary==2.9.9
  langchain==0.2.0
  pymongo==4.6.2
  python-jose==3.3.0
  ```
- Create `/public/index.html`:
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
        <p>System initialized - 07:46 AM EDT, Aug 20, 2025</p>
      </div>
      <div id="controls">
        <button id="auth" class="bg-green-500 hover:bg-green-700 text-white py-2 px-4 rounded">Authenticate</button>
      </div>
      <footer class="mt-4">
        <p>Vial MCP | Online | 2025 | v2.9</p>
      </footer>
    </div>
    <script>
      const consoleEl = document.getElementById('console');
      const log = (msg) => { consoleEl.innerHTML += `<p>${msg}</p>`; consoleEl.scrollTop = consoleEl.scrollHeight; };
      document.getElementById('auth').addEventListener('click', async () => {
        const response = await fetch('/api/auth/token', { method: 'POST', body: JSON.stringify({ username: 'admin', password: 'admin' }) });
        log(`Authentication: ${await response.text()}`);
      });
      document.addEventListener('keydown', (e) => {
        if (e.key === '/' && e.target.tagName !== 'INPUT') {
          e.preventDefault();
          log('/help - Show this help\n/auth - Authenticate with OAuth 2.0');
        }
      });
    </script>
  </body>
  </html>
  ```
- Create `vercel.json`:
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
    }
  }
  ```
- Create `package.json`:
  ```json
  {
    "name": "vial_mcp",
    "version": "2.9.0",
    "scripts": {
      "dev": "vercel dev"
    },
    "dependencies": {
      "@vercel/python": "^3.2.0"
    }
  }
  ```

## ✅ Success Criteria
- Repository initialized with `vial_mcp.vercel.app`
- Directory structure created under `/vial/`
- Docker and Vercel configurations set
- Initial files (`main.py`, `index.html`, etc.) functional
- Environment variables configured

## ⏰ Timeline Estimate
- **Duration**: 1-2 days
- **Next Phase**: Backend development (Phase 2 Guide)

## 📝 Next Steps
Proceed to **Phase 2 Guide** for API, DAO, and database implementation.