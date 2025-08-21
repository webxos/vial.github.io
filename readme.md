# Vial MCP Controller

Vial MCP Controller is a FastAPI-based application for managing AI agents, quantum synchronization, and wallet operations, compliant with Anthropic MCP standards (OAuth 2.0, JSON-RPC 2.0).

## Project Structure

```
vial/
├── public/
│   └── index.html
├── server/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── endpoints.py
│   │   ├── quantum_endpoints.py
│   │   ├── alchemist_endpoints.py
│   │   └── copilot_integration.py
│   ├── automation/
│   │   ├── auto_deploy.py
│   │   └── auto_scheduler.py
│   ├── config/
│   │   └── copilot_config.py
│   ├── models/
│   │   ├── mcp_alchemist.py
│   │   ├── webxos_wallet.py
│   │   └── auth_agent.py
│   ├── quantum/
│   │   └── quantum_sync.py
│   ├── security/
│   │   ├── security.py
│   │   └── oauth2.py
│   ├── services/
│   │   ├── database.py
│   │   ├── git_trainer.py
│   │   └── mongodb_handler.py
│   ├── sdk/
│   │   └── vial_sdk.py
│   ├── __init__.py
│   ├── error_handler.py
│   ├── logging.py
│   └── mcp_server.py
├── tests/
│   ├── test_mcp_compliance.py
│   ├── test_oauth2.py
│   ├── test_quantum_sync.py
│   ├── test_vial_monitor.py
│   ├── test_vial_sdk.py
│   └── test_mcp_alchemist_local.py
├── .env.example
├── .gitignore
├── Dockerfile
├── Dockerfile.alchemist
├── Dockerfile.monitor
├── Dockerfile.sdk
├── docker-compose.yml
├── nginx.conf
├── requirements.txt
└── readme.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vial/vial.github.io.git
   cd vial
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```

3. Build and run with Docker:
   ```bash
   docker-compose up --build
   ```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## API Endpoints

- `/auth/token`: OAuth 2.0 token endpoint
- `/jsonrpc`: JSON-RPC 2.0 endpoint with commands: `/status`, `/help`, `/train`, `/translate`, `/diagnose`, `/sync`, `/wallet`, `/monitor`, `/copilot`
- `/auth/generate-credentials`: Generate API key and secret
- `/health`: Health check endpoint

## License

MIT License