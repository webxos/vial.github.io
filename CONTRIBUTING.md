Contributing to Vial MCP Server
Thank you for contributing to the Vial MCP Server! This project supports quantum, LLM, and industrial applications with forkability to QuDAG, Cross-LLM, ServiceNow, and Alibaba Cloud.
Getting Started

Fork the repository: vial-mcp/vial-mcp-server.
Clone your fork:git clone https://github.com/your-username/vial-mcp-server.git


Install dependencies:npm install && npm ci --omit=dev
pip install -r requirements.txt


Run development environment:docker-compose up -d



Code Style

Python: Follow PEP 8, use Black (black --line-length=88).
JavaScript/TypeScript: Follow ESLint rules (npm run lint).
Run tests: pytest server/tests/ -v --cov=server --cov-report=xml.

Pull Requests

Create a branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add your feature".
Push and open a PR: git push origin feature/your-feature.
Ensure tests pass and CI completes (act -W .github/workflows/ci.yml).

Security

Use Prompt Shields for input validation.
Avoid committing sensitive data (check .gitignore).
Report vulnerabilities to security@vial.github.io.

Forkability
Contribute to forks:

QuDAG: Quantum-resistant features.
Cross-LLM: Multi-provider LLM routing.
ServiceNow/Alibaba: API integrations.

See README.md for project details and LICENSE for MIT terms.
