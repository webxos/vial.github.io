# Vial MCP Controller User Guide

## Overview
The Vial MCP Controller is a web-based platform for managing quantum-enhanced workflows, multi-LLM routing, OBS integration, and ServiceNow ticketing. This guide covers setup, usage, and key endpoints.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/webxos/vial.github.io.git
   cd vial.github.io


Install Dependencies:npm install
pip install -r requirements.txt


Configure Environment:
Copy mcp.toml.example to mcp.toml.
Set API keys and endpoints (e.g., ANTHROPIC_API_KEY, SERVICENOW_INSTANCE).


Run the Server:docker-compose up -d

Access at http://localhost:3000.

Key Endpoints

POST /mcp/quantum_rag: Execute quantum-enhanced RAG queries.curl -X POST http://localhost:3000/mcp/quantum_rag \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "test query", "quantum_circuit": "H 0; CX 0 1", "max_results": 5}'


POST /mcp/servicenow/ticket: Create ServiceNow tickets.curl -X POST http://localhost:3000/mcp/servicenow/ticket \
  -H "Authorization: Bearer <token>" \
  -d '{"short_description": "Test ticket", "description": "Test description", "urgency": "low"}'


POST /mcp/tools/obs.init: Initialize OBS scenes.curl -X POST http://localhost:3000/mcp/tools/obs.init \
  -H "Authorization: Bearer <token>" \
  -d '{"scene_name": "MyScene"}'



Usage Tips

Authentication: Obtain OAuth 2.0 tokens via /mcp/auth.
Quantum Circuits: Use QASM 2.0 format for /mcp/quantum_rag.
OBS Integration: Configure OBS_HOST, OBS_PORT, and OBS_PASSWORD in mcp.toml.
Forking: The project is forkable; see CONTRIBUTING.md.

Troubleshooting

ESLint Errors: Run npm run lint to check code style.
Test Coverage: Run pytest server/tests/ -v --cov=server.
Logs: Check logs/ for server and OBS logs.

For detailed API documentation, see docs_endpoint.py.```
