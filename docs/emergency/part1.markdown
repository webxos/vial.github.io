# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 1 (Dependencies and Structure)

**Objective**: Set up the project dependencies, directory structure, and environment configuration for the WebXOS 2025 Vial MCP SDK backend.

**Instructions for LLM**:
1. Create the project directory `webxos-vial-mcp/`.
2. Set up Python 3.11 and Node.js 18+ environments.
3. Install dependencies from the provided files.
4. Configure environment variables in `.env`.
5. Ensure compatibility with Python 3.11 and CUDA 12.1 for PyTorch.

## Step 1: Directory Structure
Create the following structure:
```
webxos-vial-mcp/
├── .github/workflows/
├── build/dockerfiles/
├── docs/emergency/backup/
├── examples/space-projects/
├── server/
│   ├── api/
│   ├── security/
│   ├── services/
│   └── tests/
├── .flake8
├── requirements-mcp.txt
├── requirements-langchain-v1.txt
├── requirements-langchain-v2.txt
├── requirements-pytorch.txt
├── package.json
```

## Step 2: Dependency Files
Create the following dependency files:

### requirements-mcp.txt
```text
fastapi==0.111.0
uvicorn==0.30.1
sqlalchemy==2.0.31
pydantic==2.7.4
qiskit==1.1.0
litellm==1.40.0
pyjwt==2.8.0
pytest==8.2.2
pytest-asyncio==0.23.7
flake8==7.1.0
coverage==7.5.4
obs-websocket-py==1.0.0
requests==2.31.0
python-jose==3.3.0
toml==0.10.2
httpx==0.27.0
```

### requirements-langchain-v1.txt
```text
langchain-openai==0.1.0
langchain-core==0.1.52
openai==1.35.7
httpx==0.27.0
```

### requirements-langchain-v2.txt
```text
langchain==0.2.0
langchain-community==0.2.0
langchain-core==0.2.38
langgraph==0.6.6
```

### requirements-pytorch.txt
```text
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
```

### package.json
```json
{
  "name": "webxos-vial-mcp",
  "version": "1.2.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint public/js/ --ext .js,.jsx,.ts,.tsx --fix"
  },
  "dependencies": {
    "next": "^14.2.5",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "next-auth": "^4.24.7",
    "jose": "^5.6.3"
  },
  "devDependencies": {
    "eslint": "^8.57.0",
    "@typescript-eslint/parser": "^6.21.0",
    "@typescript-eslint/eslint-plugin": "^6.21.0"
  }
}
```

## Step 3: Install Dependencies
```bash
pip install -r requirements-mcp.txt
pip install -r requirements-langchain-v1.txt
pip install -r requirements-langchain-v2.txt
pip install -r requirements-pytorch.txt --index-url https://download.pytorch.org/whl/cu121
npm install --legacy-peer-deps
```

## Step 4: Environment Configuration
Create `.env` with:
```text
OPENAI_API_KEY=your-openai-api-key
NASA_API_KEY=your-nasa-api-key
GITHUB_PERSONAL_ACCESS_TOKEN=your-github-pat
GITHUB_HOST=https://api.githubcopilot.com
OAUTH_CLIENT_ID=your-google-client-id
OAUTH_CLIENT_SECRET=your-google-client-secret
OAUTH_REDIRECT_URI=http://localhost:8000/mcp/auth/callback
```

## Validation
```bash
python -c "import fastapi, langchain, langchain_openai, torch; print(torch.cuda.is_available())"
npm run lint
```

**Next**: Proceed to `part2.md` for security setup.