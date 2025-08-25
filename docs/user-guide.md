WebXOS 2025 Vial MCP SDK: User Guide
Introduction
The WebXOS 2025 Vial MCP SDK is a quantum-distributed, AI-orchestrated platform for building space-related applications. It integrates Claude-Flow, OpenAI Swarm, CrewAI, SQLAlchemy, PyTorch, and APIs (NASA, SpaceX, GitHub, LangChain).
Getting Started
Prerequisites

Node.js 18+
Python 3.9+ (recommended: 3.11 or 3.12 for performance)
npm or yarn
Git
Docker and Kubernetes for deployment

Installation

Clone the repository:git clone https://github.com/webxos/webxos-vial-mcp.git
cd webxos-vial-mcp


Install Python dependencies:pip install -r requirements.txt
pip install -r requirements-pytorch.txt --index-url https://download.pytorch.org/whl/cu121


Install Node.js dependencies:npm install --legacy-peer-deps


Set up environment variables in .env:OPENAI_API_KEY=your-openai-api-key
NASA_API_KEY=your-nasa-api-key
GITHUB_PERSONAL_ACCESS_TOKEN=your-github-pat
GITHUB_HOST=https://api.githubcopilot.com
OAUTH_CLIENT_ID=your-google-client-id
OAUTH_CLIENT_SECRET=your-google-client-secret
OAUTH_REDIRECT_URI=http://localhost:8000/mcp/auth/callback
WALLET_PASSWORD=secure_wallet_password
VIAL_API_URL=http://localhost:8000
VIAL_OAUTH_REDIRECT_URI=http://localhost:3000/callback



Running the SDK

Start the backend:uvicorn server.main:app --host 0.0.0.0 --port 8000


Start the frontend:npm run build
npm run serve


Access the application at http://localhost:3000/vial.html.

Features

Wallet Management: Create and manage encrypted wallets (/mcp/wallet).
Quantum Neural Networks: Run hybrid quantum-classical models (/mcp/quantum-pytorch).
Telescope Mode: View NASA GIBS and APOD data (/mcp/telescope).
GalaxyCraft Mode: Explore a 3D galaxy (/mcp/galaxycraft).
Dropship Mode: Simulate space missions (/mcp/dropship).
RAG Queries: Retrieve and generate answers from NASA and GitHub data (/mcp/rag).

Deployment

Build Docker image:docker build -f build/dockerfiles/mcp-complete.Dockerfile -t webxos-mcp:latest .


Deploy with Helm:helm upgrade --install webxos ./helm/webxos -f helm/webxos/values.yaml


Run backup script:chmod +x scripts/backup-sdk.sh
./scripts/backup-sdk.sh



Troubleshooting

Missing requirements-pytorch.txt: Ensure the file exists or remove references to it in scripts/Dockerfile.
Missing optimizer.py: Verify server/quantum/optimizer.py is committed to the repository.
LangGraph Version Conflict: Update requirements.txt to use langgraph>=0.1.19,<0.3.0.

Support

Documentation
Community
