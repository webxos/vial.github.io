# Vial MCP Controller

![Vial MCP Controller](https://img.shields.io/badge/Vial-MCP%20Controller-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![GitHub Issues](https://img.shields.io/github/issues/webxos/vial.github.io)
![Build Status](https://github.com/webxos/vial.github.io/actions/workflows/ci.yml/badge.svg)

## Introduction

Welcome to the **Vial MCP Controller**, a groundbreaking open source platform designed for WebXOS 2025, empowering developers, researchers, and creators to orchestrate quantum-enhanced workflows, multi-LLM routing, real-time video processing, and decentralized automation through the **Model Context Protocol (MCP)**. This project is a free, community-driven tool that integrates quantum computing, artificial intelligence, and blockchain technologies into a lightweight, secure, and forkable environment. Whether you're a quantum enthusiast, AI developer, or open source contributor, the Vial MCP Controller offers a robust framework to:

- **Execute Quantum-Enhanced RAG Queries**: Leverage Qiskit and Kyber-512 for secure, quantum-powered retrieval-augmented generation.
- **Route Multi-LLM Requests**: Seamlessly integrate with Anthropic, Mistral, Google, xAI, Meta, and local LLMs using `litellm`.
- **Automate Workflows**: Create ServiceNow tickets and initialize OBS scenes for real-time SVG video rendering.
- **Visualize Topologies**: Render 3D quantum circuits and network topologies using ReactFlow and Three.js.
- **Earn DAO Rewards**: Contribute via the WebXOS 2025 VS Code extension and earn reputation in blockchain-based `.mdwallets`.

The Vial MCP Controller is designed for accessibility, security, and scalability, with a focus on minimal runtime overhead, automated data erasure for unexported wallets, and optimized error handling to prevent traffic congestion. The project is fully forkable, supported by comprehensive documentation, a custom VS Code extension, and a decentralized autonomous organization (DAO) system for community contributions.

## Purpose and Vision

The Vial MCP Controller is an open source tool that democratizes access to advanced technologies, enabling users to explore quantum computing, AI orchestration, and decentralized governance. The **Model Context Protocol (MCP)** is a novel architecture that unifies:

- **Quantum Computing**: Executes QASM 2.0 circuits with Qiskit, enhanced by Kyber-512 post-quantum cryptography for secure data processing.
- **Multi-LLM Orchestration**: Routes requests across multiple LLM providers, ensuring flexibility and redundancy.
- **Decentralized Automation**: Integrates ServiceNow for enterprise ticketing, OBS for video processing, and a blockchain-based DAO for reputation rewards.
- **3D Visualization**: Provides real-time SVG and GPU-accelerated 3D rendering for quantum circuits and network topologies.

### Why Open Source?
The project is built for the open source community to foster innovation and collaboration. By providing the **Vial SDK**, users can fork the repository, customize the MCP architecture, and contribute to features like the SVG circuit board interface and PyTorch 4x agentic scripts. The DAO system rewards contributors with reputation points stored in `.mdwallets`, incentivizing enhancements to the platform’s quantum, AI, and visualization capabilities.

### Key Benefits
- **Accessibility**: Free resources, tutorials, and sample code for learning quantum computing and AI.
- **Security**: Kyber-512 encryption, OAuth 2.0+PKCE authentication, and Prompt Shields for safe LLM interactions.
- **Scalability**: Kubernetes-ready scaling and Docker Compose deployment for production environments.
- **Community-Driven**: A DAO system where contributors earn blockchain-based rewards, ensuring transparency and fairness.

## Features

- **Quantum-RAG**: Execute quantum-enhanced retrieval-augmented generation via `/mcp/quantum_rag`.
- **Multi-LLM Routing**: Support for Anthropic, Mistral, Google, xAI, Meta, and local LLMs.
- **ServiceNow Integration**: Automate ticketing with `/mcp/servicenow/ticket`.
- **OBS Integration**: Initialize real-time video scenes with `/mcp/tools/obs.init`.
- **Monitoring**: Check system health via `/mcp/monitoring/health`.
- **Visualization**: Render quantum circuits (`QuantumCircuit.tsx`) and network topologies (`TopologyVisualizer.tsx`).
- **Authentication**: Secure `.mdwallets`-based OAuth 2.0 with automatic wallet generation and data erasure.
- **DAO Rewards**: Earn reputation for contributions, tracked on a blockchain.
- **MCP Alchemist Agent**: A PyTorch-based agent for automating quantum and LLM workflows, extensible via the Vial SDK.

## MCP Alchemist Agent

The **MCP Alchemist Agent** is a PyTorch-powered intelligent agent that automates complex workflows within the Vial MCP Controller. Built with PyTorch 4x for optimized GPU acceleration, the agent orchestrates quantum-RAG queries, LLM routing, and visualization tasks. Key functions include:

- **Quantum Workflow Automation**: Parses QASM 2.0 circuits, optimizes them with Qiskit, and executes quantum-RAG queries.
- **LLM Coordination**: Dynamically selects the best LLM provider based on query complexity and availability.
- **Visualization Enhancement**: Generates 3D-rendered SVG circuit board interfaces using Three.js, with real-time updates.
- **DAO Integration**: Tracks contributions (e.g., code submissions, circuit designs) and assigns reputation points to `.mdwallets`.

The agent is extensible via the Vial SDK, allowing contributors to add custom scripts for tasks like anomaly detection, circuit optimization, or advanced visualization. Users can test the agent locally using the WebXOS 2025 VS Code extension, which provides a sandboxed environment with a segregated SQLite database.

## Architecture Overview

The Vial MCP Controller follows a modular, quantum-distributed architecture, as shown in the following SVG diagram:

```svg
<svg width="800" height="500" viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="800" height="500" fill="#f0f0f0"/>
  <text x="400" y="30" font-size="20" text-anchor="middle" fill="#333">Vial MCP Controller Architecture</text>
  
  <!-- FastAPI Server -->
  <rect x="50" y="50" width="150" height="80" rx="10" fill="#4CAF50"/>
  <text x="125" y="90" font-size="14" text-anchor="middle" fill="#fff">FastAPI Server</text>
  <text x="125" y="110" font-size="12" text-anchor="middle" fill="#fff">/mcp/auth, /mcp/quantum_rag, etc.</text>
  
  <!-- Quantum Processor -->
  <rect x="250" y="50" width="150" height="80" rx="10" fill="#2196F3"/>
  <text x="325" y="90" font-size="14" text-anchor="middle" fill="#fff">Quantum Processor</text>
  <text x="325" y="110" font-size="12" text-anchor="middle" fill="#fff">Qiskit, Kyber-512</text>
  
  <!-- LLM Router -->
  <rect x="450" y="50" width="150" height="80" rx="10" fill="#FF9800"/>
  <text x="525" y="90" font-size="14" text-anchor="middle" fill="#fff">LLM Router</text>
  <text x="525" y="110" font-size="12" text-anchor="middle" fill="#fff">Anthropic, Mistral, xAI</text>
  
  <!-- Frontend -->
  <rect x="50" y="150" width="150" height="80" rx="10" fill="#9C27B0"/>
  <text x="125" y="190" font-size="14" text-anchor="middle" fill="#fff">React/Next.js</text>
  <text x="125" y="210" font-size="12" text-anchor="middle" fill="#fff">QuantumCircuit.tsx</text>
  
  <!-- OBS Integration -->
  <rect x="250" y="150" width="150" height="80" rx="10" fill="#FFC107"/>
  <text x="325" y="190" font-size="14" text-anchor="middle" fill="#fff">OBS Integration</text>
  <text x="325" y="210" font-size="12" text-anchor="middle" fill="#fff">obs_handler.py</text>
  
  <!-- ServiceNow -->
  <rect x="450" y="150" width="150" height="80" rx="10" fill="#F44336"/>
  <text x="525" y="190" font-size="14" text-anchor="middle" fill="#fff">ServiceNow</text>
  <text x="525" y="210" font-size="12" text-anchor="middle" fill="#fff">servicenow_endpoint.py</text>
  
  <!-- DAO and Wallets -->
  <rect x="250" y="250" width="150" height="80" rx="10" fill="#673AB7"/>
  <text x="325" y="290" font-size="14" text-anchor="middle" fill="#fff">DAO & .mdwallets</text>
  <text x="325" y="310" font-size="12" text-anchor="middle" fill="#fff">Blockchain Rewards</text>
  
  <!-- MCP Alchemist Agent -->
  <rect x="50" y="350" width="150" height="80" rx="10" fill="#009688"/>
  <text x="125" y="390" font-size="14" text-anchor="middle" fill="#fff">MCP Alchemist Agent</text>
  <text x="125" y="410" font-size="12" text-anchor="middle" fill="#fff">PyTorch 4x</text>
  
  <!-- Connections -->
  <path d="M200 90 H250" fill="none" stroke="#333" stroke-width="2"/>
  <path d="M400 90 H450" fill="none" stroke="#333" stroke-width="2"/>
  <path d="M125 130 V150" fill="none" stroke="#333" stroke-width="2"/>
  <path d="M325 130 V150" fill="none" stroke="#333" stroke-width="2"/>
  <path d="M525 130 V150" fill="none" stroke="#333" stroke-width="2"/>
  <path d="M325 230 V250" fill="none" stroke="#333" stroke-width="2"/>
  <path d="M325 330 V350" fill="none" stroke="#333" stroke-width="2"/>
</svg>

This diagram illustrates the interconnected components, with the FastAPI server acting as the central hub, coordinating quantum processing, LLM routing, frontend visualization, OBS, ServiceNow, and the DAO system.
DAO and .mdwallets System
The DAO (Decentralized Autonomous Organization) incentivizes contributions by rewarding users with reputation points stored in blockchain-based .mdwallets. These wallets serve as OAuth 2.0 keys for secure API access and are generated automatically for new users via the WebXOS 2025 VS Code extension. Key aspects include:

Wallet Generation: New users run WebXOS: Initialize Wallet to create a .mdwallet file with a Kyber-512 keypair and JWT token.
Reputation Rewards: Contributions (e.g., code, circuit designs, PyTorch scripts) earn points, recorded on a blockchain (e.g., Ethereum or a custom chain).
Data Erasure: Unexported wallets are deleted when the VS Code terminal closes, ensuring a lightweight runtime.
Transparency: Blockchain ensures immutable tracking of contributions and rewards.

Contributors can design custom MCP SVG circuit board interfaces or enhance 3D GPU rendering, earning higher reputation for impactful submissions. The DAO operates via smart contracts, with details in docs/dao.md (to be added in future phases).
Setup Guide for New Users
Prerequisites

Node.js: v20 LTS
Python: 3.11
Docker: Latest version
VS Code: 1.85.0+
Git: For cloning and forking

Installation

Clone the Repository:git clone https://github.com/webxos/vial.github.io.git
cd vial.github.io


Install Dependencies:npm install
pip install -r requirements.txt


Configure Environment:
Copy mcp.toml.example to mcp.toml.
Set API keys (e.g., ANTHROPIC_API_KEY, SERVICENOW_INSTANCE, JWT_SECRET).
Example mcp.toml:[server]
host = "0.0.0.0"
port = 3000
[auth]
JWT_SECRET = "your-secret-key"
[llm]
ANTHROPIC_API_KEY = "your-key"
[obs]
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "your-password"
[servicenow]
SERVICENOW_INSTANCE = "your-instance"




Run the Server:docker-compose up -d

Access at http://localhost:3000.

Installing WebXOS 2025 VS Code Extension

Download Extension:
Get webxos-2025.vsix from extensions/webxos-2025/.


Install in VS Code:code --install-extension extensions/webxos-2025/webxos-2025.vsix


Configure Extension:
Set webxos-2025.apiEndpoint to http://localhost:3000 in VS Code settings.
Set webxos-2025.walletDir to ~/.webxos/wallets.



Using the Extension

Initialize Wallet:
Run WebXOS: Initialize Wallet in the Command Palette.
A .mdwallet file is created in ~/.webxos/wallets.
Export with WebXOS: Export Wallet to persist it.


Run Commands:
WebXOS: Run Quantum RAG Query: Execute quantum-RAG queries.
WebXOS: Create ServiceNow Ticket: Create tickets.
WebXOS: Initialize OBS Scene: Initialize OBS scenes.
WebXOS: Check System Health: Monitor server status.



Forking the Repository
To contribute or customize the project:

Fork the Repo:
Click "Fork" on github.com/webxos/vial.github.io.


Clone Your Fork:git clone https://github.com/<your-username>/vial.github.io.git
cd vial.github.io


Create a Branch:git checkout -b feature/your-feature


Make Changes:
Add code, tests, or documentation.
Run npm run lint and flake8 server/ to ensure style compliance.
Run pytest server/tests/ -v --cov=server for 90%+ coverage.


Commit and Push:git commit -m "Add your-feature"
git push origin feature/your-feature


Submit a Pull Request:
Create a PR against main on the original repo.
Include tests and update docs/user_guide.md if needed.



API Access and Tools
The Vial MCP Controller exposes a FastAPI-based API for interacting with its features. All endpoints require OAuth 2.0 authentication via .mdwallets.
Endpoints

POST /mcp/auth: Authenticate with a .mdwallet.curl -X POST http://localhost:3000/mcp/auth \
  -d '{"wallet_id": "your_wallet_id", "public_key": "your_public_key"}'


POST /mcp/quantum_rag: Run quantum-RAG queries.curl -X POST http://localhost:3000/mcp/quantum_rag \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "test query", "quantum_circuit": "H 0; CX 0 1", "max_results": 5}'


POST /mcp/servicenow/ticket: Create ServiceNow tickets.curl -X POST http://localhost:3000/mcp/servicenow/ticket \
  -H "Authorization: Bearer <token>" \
  -d '{"short_description": "Test ticket", "description": "Test description", "urgency": "low"}'


POST /mcp/tools/obs.init: Initialize OBS scenes.curl -X POST http://localhost:3000/mcp/tools/obs.init \
  -H "Authorization: Bearer <token>" \
  -d '{"scene_name": "MyScene"}'


GET /mcp/monitoring/health: Check system health.curl -H "Authorization: Bearer <token>" http://localhost:3000/mcp/monitoring/health



Tools

Vial SDK: A Python library (@modelcontextprotocol/sdk) for interacting with MCP endpoints. Install with:pip install @modelcontextprotocol/sdk

Example usage:from modelcontextprotocol import MCPClient
client = MCPClient(api_endpoint="http://localhost:3000", token="your_token")
response = client.quantum_rag(query="test", circuit="H 0; CX 0 1", max_results=5)


WebXOS 2025 VS Code Extension: Provides a GUI for API access, wallet management, and real-time visualization.
Postman Collection: Available in docs/postman_collection.json for testing endpoints.
Swagger UI: Access at http://localhost:3000/docs when the server is running.

Resources

Quantum Computing: Qiskit Tutorials
LLM Integration: Litellm Documentation
FastAPI: FastAPI Documentation
ReactFlow: ReactFlow Documentation
Three.js: Three.js Documentation
PyTorch: PyTorch Tutorials
DAO and Blockchain: Ethereum Developer Resources

Directory Structure
Below is the complete directory tree of the vial.github.io repository, with descriptions of each file and folder:
vial.github.io/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow for CI/CD, linting, and testing
├── docs/
│   ├── user_guide.md           # User guide for setup, usage, and extension
│   ├── CONTRIBUTING.md         # Contribution guidelines for forking and PRs
│   └── postman_collection.json # Postman collection for API testing
├── extensions/
│   └── webxos-2025/
│       ├── package.json        # VS Code extension metadata and commands
│       └── src/
│           └── extension.ts    # Extension logic for wallet management and API calls
├── scripts/
│   └── update_deps.sh          # Script for updating Node.js and Python dependencies
├── server/
│   ├── api/
│   │   ├── auth_endpoint.py    # /mcp/auth endpoint for .mdwallets authentication
│   │   ├── quantum_rag_endpoint.py # /mcp/quantum_rag endpoint for quantum-RAG queries
│   │   ├── servicenow_endpoint.py  # /mcp/servicenow/ticket endpoint for ticketing
│   │   └── monitoring_endpoint.py  # /mcp/monitoring/health endpoint for system health
│   ├── config/
│   │   ├── settings.py        # Configuration loader for mcp.toml
│   │   └── scaling_config.py  # Kubernetes HPA configuration
│   ├── services/
│   │   ├── llm_router.py      # Multi-LLM routing with litellm
│   │   ├── quantum_processor.py # Quantum circuit execution with Qiskit
│   │   └── obs_handler.py     # OBS WebSocket integration for video scenes
│   ├── tests/
│   │   ├── test_llm_router.py # Tests for llm_router.py
│   │   ├── test_quantum_rag_endpoint.py # Tests for quantum_rag_endpoint.py
│   │   ├── test_quantum_processor.py   # Tests for quantum_processor.py
│   │   ├── test_servicenow_endpoint.py # Tests for servicenow_endpoint.py
│   │   ├── test_monitoring_endpoint.py # Tests for monitoring_endpoint.py
│   │   └── test_auth_endpoint.py      # Tests for auth_endpoint.py
│   └── main.py                # FastAPI main entry point
├── typescript/
│   ├── src/
│   │   ├── components/
│   │   │   ├── QuantumCircuit.tsx     # React component for quantum circuit visualization
│   │   │   └── TopologyVisualizer.tsx # React component for network topology visualization
│   │   ├── tests/
│   │   │   └── TopologyVisualizer.test.tsx # Tests for TopologyVisualizer.tsx
│   │   └── utils/
│   │       └── fetcher.ts            # Utility for secure API calls with OAuth
├── .eslintrc.json                    # ESLint configuration for JavaScript/TypeScript
├── docker-compose.yml                # Docker Compose for production deployment
├── mcp.toml.example                  # Example configuration file
├── package.json                      # Node.js dependencies and scripts
├── requirements.txt                  # Python dependencies
└── README.md                         # This file

File Descriptions

.github/workflows/ci.yml: Automates linting, testing, and deployment with GitHub Actions.
docs/user_guide.md: Detailed setup and usage instructions, including VS Code extension.
docs/CONTRIBUTING.md: Guidelines for contributing and earning DAO rewards.
extensions/webxos-2025/package.json: Metadata for the WebXOS 2025 VS Code extension.
extensions/webxos-2025/src/extension.ts: Logic for wallet management and API interactions.
scripts/update_deps.sh: Automates dependency updates with security checks.
server/api/: FastAPI endpoints for authentication, quantum-RAG, ServiceNow, and monitoring.
server/config/: Configuration and scaling settings for the server.
server/services/: Core services for LLM routing, quantum processing, and OBS integration.
server/tests/: Pytest suite for 90%+ coverage of server components.
server/main.py: FastAPI entry point with global error handling.
typescript/src/components/: React components for visualization.
typescript/src/tests/: Jest tests for frontend components.
typescript/src/utils/fetcher.ts: Secure API fetcher with OAuth 2.0.
docker-compose.yml: Production deployment configuration.
mcp.toml.example: Template for environment configuration.
package.json: Node.js dependencies and scripts for linting/testing.
requirements.txt: Python dependencies for the server.

MCP SVG Circuit Board Interface
The MCP SVG Circuit Board Interface is a customizable visualization layer for the MCP architecture, rendered as an interactive SVG diagram. Users can contribute designs via the Vial SDK or VS Code extension, integrating with QuantumCircuit.tsx and TopologyVisualizer.tsx. The following SVG illustrates a sample circuit board layout:
<svg width="600" height="400" viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="600" height="400" fill="#1a1a1a"/>
  <text x="300" y="30" font-size="18" text-anchor="middle" fill="#fff">MCP Circuit Board Interface</text>
  
  <!-- Quantum Circuit -->
  <rect x="50" y="50" width="100" height="60" rx="5" fill="#2196F3"/>
  <text x="100" y="80" font-size="12" text-anchor="middle" fill="#fff">Quantum Module</text>
  
  <!-- LLM Router -->
  <rect x="200" y="50" width="100" height="60" rx="5" fill="#FF9800"/>
  <text x="250" y="80" font-size="12" text-anchor="middle" fill="#fff">LLM Router</text>
  
  <!-- DAO Rewards -->
  <rect x="350" y="50" width="100" height="60" rx="5" fill="#673AB7"/>
  <text x="400" y="80" font-size="12" text-anchor="middle" fill="#fff">DAO Rewards</text>
  
  <!-- Connections -->
  <path d="M150 80 H200" fill="none" stroke="#fff" stroke-width="2"/>
  <path d="M300 80 H350" fill="none" stroke="#fff" stroke-width="2"/>
  <circle cx="150" cy="80" r="5" fill="#fff"/>
  <circle cx="300" cy="80" r="5" fill="#fff"/>
</svg>

Contributors can enhance this interface by submitting SVG designs or PyTorch scripts for 3D GPU rendering, earning DAO rewards.
3D GPU Rendering and PyTorch 4x Agentic Scripts
The Vial MCP Controller uses Three.js for 3D visualization and PyTorch 4x for agentic scripts, enabling GPU-accelerated rendering of quantum circuits and topologies. The MCP Alchemist Agent leverages PyTorch to:

Optimize quantum circuits for visualization.
Generate dynamic 3D models of network topologies.
Automate LLM query routing based on performance metrics.

Contributors can extend these scripts via the Vial SDK, adding features like real-time circuit animation or anomaly detection. Example PyTorch script structure (to be added in server/services/alchemist_agent.py in future phases):
import torch
class AlchemistAgent:
    def __init__(self):
        self.model = torch.nn.Module()  # Placeholder for 4x-optimized model
    def optimize_circuit(self, circuit: str) -> str:
        # Optimize QASM circuit with GPU
        pass

Contributing and DAO Rewards
To contribute:

Fork and Clone: Follow the forking guide above.
Develop Features:
Add endpoints in server/api/.
Enhance visualizations in typescript/src/components/.
Create PyTorch scripts for the MCP Alchemist Agent.


Test and Lint:npm run lint
flake8 server/
pytest server/tests/ -v --cov=server


Submit PRs: Include tests and documentation updates.
Earn Rewards: Contributions earn reputation points in .mdwallets, tracked on the blockchain.

See CONTRIBUTING.md for detailed guidelines.
Troubleshooting

ESLint Errors: Run npm run lint to fix code style issues.
Flake8 Errors: Run flake8 server/ to ensure Python style compliance.
Test Failures: Run pytest server/tests/ -v and check logs in logs/.
Extension Issues: Verify webxos-2025.apiEndpoint in VS Code settings.
API Errors: Check http://localhost:3000/docs for endpoint details.

License
This project is licensed under the MIT License. See LICENSE for details.

Thank you for joining the Vial MCP Controller community! Fork, build, and contribute to shape the future of quantum-AI automation.```
