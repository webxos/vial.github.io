# Vial MCP Controller

A production-grade MCP agent orchestrating 4x vial agents with FastAPI, PyTorch, SQLAlchemy, Qiskit, and Next.js.

## MCP Server

![MCP Manifest](https://img.shields.io/badge/MCP%20Manifest-mcp.toml-blue)
![Run MCP Locally](https://img.shields.io/badge/Run%20MCP%20Locally-scripts/mcp_dev.sh-green)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/vial/vial-mcp.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   npm install
   ```
3. Set environment variables:
   ```bash
   export JWT_SECRET=your_jwt_secret
   export DATABASE_URL=postgresql+psycopg://user:pass@localhost/db
   export REDIS_URL=redis://localhost:6379
   export VERCEL_API_TOKEN=your_vercel_token
   export VERCEL_PROJECT_ID=your_project_id
   export GITHUB_TOKEN=your_github_token
   export QISKIT_IBM_TOKEN=your_ibm_token
   export MODEL_PROVIDER=anthropic
   export ANTHROPIC_API_KEY=your_anthropic_key
   export XAI_API_KEY=your_xai_key
   ```
4. Run MCP server locally:
   ```bash
   ./scripts/mcp_dev.sh
   ```
5. Deploy to Vercel:
   ```bash
   vercel --prod
   ```

### Features

- MCP server for Cursor/Claude Desktop integration
- OAuth2 authentication with scoped JWTs
- Qiskit quantum circuit building
- Next.js frontend with SVG editor
- Prometheus metrics and audit logging
