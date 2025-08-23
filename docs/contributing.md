Contributing to Vial MCP
Overview
The Vial MCP project welcomes contributions to enhance its quantum, LLM, and industrial capabilities. Fork the repo to QuDAG, Cross-LLM, ServiceNow, or Alibaba Cloud MCP servers.
Forking and PR Process

Fork the repository: git clone https://github.com/your-username/vial-mcp-server.git.
Create a branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add feature X".
Push and create PR: git push origin feature/your-feature.

Coding Standards

Use Python 3.11+, TypeScript for frontend.
Follow PEP 8 and Flake8 (max line length: 88).
Use Pydantic for models, async/await for API calls.

Testing Requirements

Achieve 90%+ coverage with pytest.
Test all endpoints, tools, and integrations.
Run pytest server/tests/ -v --cov=server --cov-report=xml.

Security Guidelines

Validate inputs with Microsoft Prompt Shields.
Use OAuth 2.0+PKCE for authentication.
Encrypt sensitive data with AES-256/Kyber-512.

Community
Contribute to the MCP registry: bash scripts/publish_registry.sh.
