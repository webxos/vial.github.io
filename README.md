# Vial MCP Controller

A modular control plane for AI-driven task management with visual API routing, quantum synchronization, and WebXOS wallet integration.

## Project Structure
```
vial_mcp/
├── public/
│   ├── index.html
│   └── js/
│       └── threejs_integrations.js
├── scripts/
│   ├── deploy_script.py
│   └── deployment_validator.py
├── server/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── cache_control.py
│   │   ├── comms_hub.py
│   │   ├── copilot_integration.py
│   │   ├── endpoints.py
│   │   ├── health_check.py
│   │   ├── help.py
│   │   ├── jsonrpc.py
│   │   ├── middleware.py
│   │   ├── quantum_endpoints.py
│   │   ├── rate_limiter.py
│   │   ├── stream.py
│   │   ├── troubleshoot.py
│   │   ├── upload.py
│   │   ├── visual_router.py
│   │   ├── void.py
│   │   ├── websocket.py
│   │   ├── webxos_wallet.py
│   │   └── webxos_wallet_engine.py
│   ├── automation/
│   │   ├── __init__.py
│   │   ├── deployment.py
│   │   └── task_scheduler.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── final_config.py
│   │   └── settings.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── auth_agent.py
│   │   ├── mcp_alchemist.py
│   │   ├── visual_components.py
│   │   └── webxos_wallet.py
│   ├── quantum/
│   │   ├── __init__.py
│   │   └── quantum_sync.py
│   ├── sdk/
│   │   ├── __init__.py
│   │   └── vial_sdk.py
│   ├── security/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── rbac.py
│   │   └── security_headers.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── advanced_logging.py
│   │   ├── agent_tasks.py
│   │   ├── backup_restore.py
│   │   ├── database.py
│   │   ├── error_recovery.py
│   │   ├── git_trainer.py
│   │   ├── prompt_training.py
│   │   ├── training_scheduler.py
│   │   └── vial_manager.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── load_test_training_scheduler.py
│   │   ├── test_agent_tasks.py
│   │   ├── test_copilot_integrations.py
│   │   ├── test_deployment_vps.py
│   │   ├── test_integration.py
│   │   ├── test_prompt_training.py
│   │   └── test_security.py
│   ├── error_handler.py
│   ├── logging.py
│   ├── logging_config.py
│   └── mcp_server.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

## SVG Diagram
Below is an example SVG diagram of the Vial MCP architecture:

```svg
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
    <rect x="50" y="50" width="100" height="50" fill="green" stroke="black"/>
    <text x="60" y="80" fill="white">API Endpoint</text>
    <rect x="200" y="150" width="100" height="50" fill="green" stroke="black"/>
    <text x="210" y="180" fill="white">LLM Model</text>
    <rect x="350" y="250" width="100" height="50" fill="green" stroke="black"/>
    <text x="360" y="280" fill="white">Database</text>
    <line x1="100" y1="100" x2="200" y2="150" stroke="white" stroke-width="2"/>
    <line x1="250" y1="200" x2="350" y2="250" stroke="white" stroke-width="2"/>
</svg>
```

## Setup Instructions
1. **Clone Repository**:
   ```bash
   git clone https://github.com/yourusername/vial.github.io.git
   cd vial.github.io
   ```
2. **Setup Environment**:
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```
4. **Run Locally**:
   ```bash
   uvicorn server.mcp_server:app --reload --host 0.0.0.0 --port 8000
   ```
5. **Test**:
   ```bash
   pytest server/tests/ -v
   flake8 . --max-line-length=88 --extend-ignore=E203
   ```
6. **Deploy with Docker**:
   ```bash
   docker-compose up --build -d
   ```

## Features
- **Visual API Router**: Drag-and-drop components with Three.js visualization.
- **WebXOS Wallet**: Export/import, DAO creation, staking, and multi-signature transactions.
- **Quantum Sync**: Visual quantum circuit generation.
- **PyTorch Agents**: 4 trainable agents with SQLAlchemy persistence.
- **Dockerized Deployment**: Fully containerized with MongoDB and Redis support.

## Troubleshooting
- **Import Errors**: Ensure `__init__.py` files exist in all directories.
- **Database Issues**: Run `init_db()` on startup.
- **Three.js Errors**: Verify CDN URLs and WASM support.
- **Docker Issues**: Check Dockerfile paths and `.env` settings.

## Quality Gates
- Frontend: 95%+ test coverage, 90+ Lighthouse score, WCAG 2.1 AA compliance.
- Backend: 90%+ test coverage, <200ms API response time, <512MB memory usage.
- Integration: 100% end-to-end test pass rate, handles 1000 concurrent users.
