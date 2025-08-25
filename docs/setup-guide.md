Setup Guide for WebXOS 2025 Vial MCP SDK
This guide covers the setup and configuration of the WebXOS 2025 Vial MCP SDK, including fixes for common issues.
Prerequisites

Python 3.11
Node.js 18+
Docker
Git

Installation

Clone the Repository:
git clone https://github.com/webxos/webxos-vial-mcp.git
cd webxos-vial-mcp


Install Python Dependencies:
pip install -r requirements-mcp.txt


Install Node.js Dependencies:
npm install --legacy-peer-deps


Configure Environment Variables:Create a .env file:
NASA_API_KEY=your-nasa-api-key
GITHUB_PERSONAL_ACCESS_TOKEN=your-github-pat
GITHUB_HOST=https://api.githubcopilot.com
OAUTH_CLIENT_ID=your-google-client-id
OAUTH_CLIENT_SECRET=your-google-client-secret
OAUTH_REDIRECT_URI=http://localhost:8000/mcp/auth/callback



CI Pipeline Setup

Permissions: The CI workflow includes issues: write for GitHub integration.
Linting: Uses flake8 for Python code (configured in .flake8).
Artifacts: Uses actions/upload-artifact@v4.

Troubleshooting

LangGraph Error: Ensure langgraph==0.6.6 in requirements-mcp.txt.
Linting: Run flake8 server/ examples/ to verify code quality.
CI Failures: Check GitHub Action logs and ensure all environment variables are set.

Running the Server
uvicorn server.main:app --host 0.0.0.0 --port 8000

Next Steps

Explore docs/tutorials/security-guide.md for security setup.
Run example projects in examples/space-projects/.
