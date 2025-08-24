# Contributing to Vial MCP Controller

Thank you for contributing to the Vial MCP Controller! This guide outlines how to contribute effectively.

## Getting Started
1. **Fork the Repository**:
   ```bash
   git clone https://github.com/<your-username>/vial.github.io.git
   cd vial.github.io


Set Up Environment:
Install Node.js (v20) and Python (3.11).
Run npm install and pip install -r requirements.txt.
Copy mcp.toml.example to mcp.toml and configure settings.


Run Locally:docker-compose up -d

Access at http://localhost:3000.

Development Guidelines

Code Style:
JavaScript/TypeScript: Run npm run lint (ESLint).
Python: Run flake8 server/ --max-line-length=88 --extend-ignore=E203,W503.


Testing:
Write tests in server/tests/ using pytest.
Ensure 90%+ coverage: pytest server/tests/ -v --cov=server.


Commits:
Use clear commit messages (e.g., Add quantum_rag_endpoint.py for RAG queries).
Follow Conventional Commits.


Pull Requests:
Create PRs against main.
Include tests and update docs/user_guide.md if needed.
Pass CI checks (ci.yml).



Key Components

Backend: FastAPI server in server/ with endpoints like /mcp/quantum_rag, /mcp/servicenow/ticket.
Frontend: React/Next.js in typescript/src/ with components like QuantumCircuit.tsx.
Quantum: Qiskit-based processing in server/services/quantum_processor.py.
OBS: WebSocket integration in server/services/obs_handler.py.

Running Tests
npm run lint
flake8 server/
pytest server/tests/ -v --cov=server --cov-report=xml

Updating Dependencies
./scripts/update_deps.sh

Issues and Feature Requests

Check issues before submitting.
Use clear titles and describe the issue or feature in detail.

License
This project is licensed under the MIT License. See LICENSE for details.```
