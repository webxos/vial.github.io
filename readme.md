# Vial MCP Controller

The Vial MCP Controller is a FastAPI-based server for managing AI-driven tasks, integrating with GitHub, MongoDB, Redis, and quantum computing via Qiskit. It supports OAuth 2.0 and JSON-RPC 2.0 for secure and standardized communication.

## Project Structure

```
vial.github.io/
├── server/
│   ├── api/
│   │   ├── auth.py
│   │   ├── cache_control.py
│   │   ├── copilot_integration.py
│   │   ├── docs.py
│   │   ├── endpoints.py
│   │   ├── health_check.py
│   │   ├── middleware.py
│   │   ├── quantum_endpoints.py
│   │   ├── rate_limiter.py
│   │   └── websocket.py
│   ├── automation/
│   │   ├── auto_deploy.py
│   │   └── auto_scheduler.py
│   ├── models/
│   │   ├── auth_agent.py
│   │   ├── mcp_alchemist.py
│   │   └── webxos_wallet.py
│   ├── quantum/
│   │   └── quantum_sync.py
│   ├── services/
│   │   ├── audit_log.py
│   │   ├── backup_restore.py
│   │   ├── database.py
│   │   ├── git_trainer.py
│   │   ├── mongodb_handler.py
│   │   ├── notification.py
│   │   └── redis_handler.py
│   ├── tests/
│   │   ├── test_auto_deploy.py
│   │   ├── test_backup_restore.py
│   │   ├── test_docs.py
│   │   ├── test_notification.py
│   │   ├── test_oauth2.py
│   │   ├── test_quantum_sync.py
│   │   └── test_redis_handler.py
│   ├── config.py
│   ├── error_handler.py
│   ├── logging.py
│   ├── mcp_server.py
│   ├── security.py
│   ├── utils.py
│   └── sdk/
│       └── vial_sdk.py
├── public/
│   ├── index.html
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css
│   │   └── js/
│   │       └── main.js
├── .gitignore
├── fullbuildguide.md
├── pyproject.toml
├── requirements.txt
├── setup.py
└── README.md
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vial.github.io.git
   cd vial.github.io
   ```

2. **Install Dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   Create a `.env` file in the project root with:
   ```
   GITHUB_TOKEN=your_github_token
   GITHUB_USERNAME=your_github_username
   MONGO_URL=mongodb://localhost:27017
   REDIS_URL=redis://localhost:6379
   NOTIFICATION_API_URL=https://api.example.com/notify
   ```

4. **Run the Server**:
   ```bash
   uvicorn server.mcp_server:app --host 0.0.0.0 --port 8000
   ```

5. **Run Tests**:
   ```bash
   pytest server/tests/
   ```

## API Endpoints

- `/health`: Check server health.
- `/jsonrpc`: JSON-RPC 2.0 endpoint for commands (status, help).
- `/auth/token`: OAuth 2.0 token generation.
- `/auth/generate-credentials`: Generate API key and secret.
- `/docs`: API documentation (Swagger UI).
- `/redoc`: API documentation (ReDoc).

## License

MIT License
