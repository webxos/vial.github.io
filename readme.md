# Vial MCP Controller

## Repository Structure

```
vial/
├── .env.example                    # Environment variable template
├── .gitignore                     # Git ignore rules
├── docker-compose.yml             # Multi-container orchestration
├── docker-compose.prod.yml        # Production configuration
├── Dockerfile                     # Main application container
├── Dockerfile.alchemist           # Alchemist agent container
├── Dockerfile.monitor             # Monitoring container
├── Dockerfile.sdk                 # SDK container
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── Makefile                       # Build automation
├── public/
│   └── index.html                 # 3D interface and terminal
├── server/
│   ├── __init__.py
│   ├── mcp_server.py             # FastAPI application entry
│   ├── security.py               # Security utilities
│   ├── config.py                 # Configuration management
│   ├── error_handler.py          # Error handling middleware
│   ├── logging.py                # Logging configuration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py               # Authentication endpoints
│   │   ├── endpoints.py          # Main MCP endpoints
│   │   ├── alchemist_endpoints.py # Alchemist agent endpoints
│   │   ├── monitor_endpoints.py  # Monitoring endpoints
│   │   ├── quantum_endpoints.py  # Quantum sync endpoints
│   │   ├── sdk_endpoints.py      # SDK endpoints
│   │   └── copilot_integration.py # Copilot integration
│   ├── config/
│   │   ├── __init__.py
│   │   └── copilot_config.py     # Copilot configuration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── alchemy_pytorch.py    # PyTorch alchemist models
│   │   ├── auth_agent.py         # Authentication agent
│   │   ├── mcp_alchemist.py      # MCP alchemist logic
│   │   └── webxos_wallet.py      # Wallet management
│   ├── sdk/
│   │   ├── __init__.py
│   │   ├── vial_monitor.py       # Monitoring SDK
│   │   └── vial_sdk.py           # SDK core
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database.py           # Database initialization
│   │   ├── git_trainer.py        # Git-based training
│   │   └── mongodb_handler.py    # MongoDB handler
│   ├── automation/
│   │   ├── __init__.py
│   │   ├── auto_deploy.py        # Automated deployment
│   │   └── auto_scheduler.py     # Automated scheduling
│   ├── security/
│   │   ├── __init__.py
│   │   └── oauth2.py             # OAuth 2.0 implementation
│   ├── quantum/
│   │   ├── __init__.py
│   │   └── quantum_sync.py       # Quantum synchronization
│   └── utils.py                  # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_mcp_alchemist_local.py
│   ├── test_mcp_compliance.py
│   ├── test_oauth2.py
│   ├── test_quantum_sync.py
│   ├── test_vial_monitor.py
│   └── test_vial_sdk.py
└── ssl/
    ├── cert.pem                   # SSL certificate
    └── key.pem                   # SSL key
```

## Detailed Component Breakdown
1. **Frontend (public/index.html)**:
   - Features: 3D visualization with Three.js, terminal interface for commands (/help, /login, /train, etc.), OAuth 2.0 authentication prompt.
   - Dependencies: Three.js and Tailwind CSS via CDN.
   - Usage: Access at http://localhost:8000 post-deployment, authenticate with admin/admin, and use terminal commands.

2. **Backend (server/)**
   - **MCP Server (mcp_server.py)**: FastAPI-based server with JSON-RPC 2.0, handling endpoints like `/jsonrpc`, `/auth/token`, and `/auth/generate-credentials`.
   - **API Endpoints**:
     - `endpoints.py`: Core JSON-RPC endpoints.
     - `auth.py`: OAuth 2.0 login.
     - `quantum_endpoints.py`: Quantum sync and wallet updates.
     - `alchemist_endpoints.py`: Alchemist training/translation.
     - `sdk_endpoints.py`: SDK deployment.
     - `monitor_endpoints.py`: System monitoring.
     - `copilot_integration.py`: Copilot features.
   - **Services**:
     - `database.py`: SQLite and MongoDB initialization.
     - `mongodb_handler.py`: MongoDB storage.
     - `git_trainer.py`: Git-based training.
   - **Security (security.py, security/oauth2.py)**: OAuth 2.0 with JWT (HS256).
   - **Utils (utils.py)**: JSON-RPC and GitHub utilities.

3. **Docker Images**:
   - `Dockerfile`: Main application with FastAPI, SQLite, MongoDB, and Redis support.
   - `Dockerfile.alchemist`: Alchemist-specific image with PyTorch and MongoDB.
   - `Dockerfile.monitor`: Monitoring image with `psutil` and `docker`.
   - `Dockerfile.sdk`: SDK image for deployment.
   - `docker-compose.yml`: Orchestrates FastAPI, MongoDB, Redis, and Nginx.

4. **Tests (tests/)**:
   - Validates OAuth, quantum sync, alchemist, SDK, and monitor functionality with FastAPI TestClient.

## Step-by-Step Rebuild Instructions
1. **Rebuild from Scratch**:
   - Clone the repo and set up as per Installation steps.
   - Copy `.env.example` to `.env` and fill in secrets.
   - Build and run: `docker-compose up -d`.

2. **Verify Components**:
   - **Authentication**: `curl -X POST http://localhost:8000/auth/token -d '{"username": "admin", "password": "admin"}' -H "Content-Type: application/json"`.
   - **JSON-RPC**: `curl -X POST http://localhost:8000/jsonrpc -d '{"jsonrpc": "2.0", "method": "status", "id": 1}' -H "Content-Type: application/json"`.
   - **Health Check**: `curl http://localhost:8000/health`.

3. **Automate and Monitor**:
   - Start `auto_scheduler.py` and `auto_deploy.py` via Docker entrypoints.
   - Monitor logs for sync, training, and deployment.

## Specifications
- **Languages**: Python, HTML, JavaScript
- **Frameworks**: FastAPI, Three.js
- **Libraries**: PyTorch, Qiskit, Octokit, MongoDB, Redis, SQLAlchemy
- **Security**: OAuth 2.0 with HS256 JWT, HTTPS via Nginx
- **Scalability**: Dockerized, multi-container architecture
- **Performance**: CUDA support for PyTorch

## Troubleshooting
- **Authentication Issues**: Verify `JWT_SECRET` in `.env`.
- **Docker Errors**: Ensure Docker daemon runs and port 8000 is free.
- **MongoDB/Redis**: Check connection URLs in `.env`.

## Future Enhancements
- Expand quantum simulations with Qiskit.
- Integrate additional LLM frameworks (Anthropic, xAI).
- Enhance Copilot features for code generation.