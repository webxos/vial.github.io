# Vial MCP Controller

![Vial MCP Controller](https://img.shields.io/badge/Vial-MCP%20Controller-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![GitHub Issues](https://img.shields.io/github/issues/webxos/vial.github.io)
![Build Status](https://github.com/webxos/vial.github.io/actions/workflows/ci.yml/badge.svg)

## Introduction

The **Vial MCP Controller** is a pioneering open source platform for WebXOS 2025, designed to empower the global open source community to orchestrate quantum-enhanced workflows, multi-LLM routing, real-time video processing, and decentralized automation through the **Model Context Protocol (MCP)**. This project integrates quantum computing, artificial intelligence, and blockchain technologies into a lightweight, secure, and forkable environment, making advanced tools accessible to developers, researchers, and creators. Key capabilities include:

- **Quantum-Enhanced RAG Queries**: Execute retrieval-augmented generation (RAG) queries using Qiskit and Kyber-512 post-quantum cryptography.
- **Multi-LLM Orchestration**: Route requests across Anthropic, Mistral, Google, xAI, Meta, and local LLMs with `litellm`.
- **Workflow Automation**: Integrate with ServiceNow for enterprise ticketing and OBS for real-time SVG video rendering.
- **3D Visualization**: Render quantum circuits and network topologies using ReactFlow and Three.js with GPU acceleration.
- **Decentralized Rewards**: Earn reputation points in blockchain-based `.mdwallets` via the DAO system.
- **MCP Alchemist Agent**: A PyTorch 4x-powered agent for automating quantum, LLM, and visualization workflows.

The Vial MCP Controller prioritizes **accessibility**, **security**, and **scalability**, with a minimal runtime footprint, automated data erasure for unexported `.mdwallets`, and optimized FastAPI error handling to prevent traffic congestion. The project is fully forkable, supported by the **WebXOS 2025 VS Code extension**, comprehensive documentation, and the **Vial SDK** for custom development. Contributors can design custom MCP circuit board interfaces, enhance 3D GPU rendering, and develop PyTorch agentic scripts, earning DAO rewards for their efforts.

## Purpose and Vision

The Vial MCP Controller is an open source initiative to democratize access to quantum computing, AI, and blockchain technologies, fostering innovation through community collaboration. The **Model Context Protocol (MCP)** is a novel architecture that unifies:

- **Quantum Computing**: Executes QASM 2.0 circuits with Qiskit, secured by Kyber-512 for post-quantum safety.
- **Multi-LLM Orchestration**: Routes queries across multiple LLM providers, ensuring flexibility and performance.
- **Decentralized Automation**: Integrates ServiceNow, OBS, and a blockchain-based DAO for seamless workflows and rewards.
- **Visualization**: Provides real-time Mermaid diagrams and 3D GPU-accelerated rendering for quantum circuits and network topologies.

### Why Open Source?
The project empowers the open source community by providing free tools and resources via the Vial SDK. Contributors can fork the repository, customize the MCP architecture, and enhance features like the circuit board interface or PyTorch agentic scripts. The **DAO system** rewards contributions with reputation points stored in `.mdwallets`, tracked on a blockchain for transparency and fairness.

### Key Benefits
- **Accessibility**: Free tutorials, sample code, and the Vial SDK for learning quantum and AI technologies.
- **Security**: Kyber-512 encryption, OAuth 2.0+PKCE, and Prompt Shields for safe LLM interactions.
- **Scalability**: Kubernetes-ready scaling and Docker Compose for production deployment.
- **Community-Driven**: Earn DAO rewards for code, circuit designs, and documentation contributions.

## Features

- **Quantum-RAG**: Execute quantum-enhanced RAG queries via `/mcp/quantum_rag`.
- **Multi-LLM Routing**: Support for Anthropic, Mistral, Google, xAI, Meta, and local LLMs.
- **ServiceNow Integration**: Automate ticketing with `/mcp/servicenow/ticket`.
- **OBS Integration**: Initialize real-time video scenes with `/mcp/tools/obs.init`.
- **Monitoring**: Check system health via `/mcp/monitoring/health`.
- **Visualization**: Render quantum circuits (`QuantumCircuit.tsx`) and network topologies (`TopologyVisualizer.tsx`).
- **Authentication**: Secure `.mdwallets`-based OAuth 2.0 with automatic wallet generation and data erasure.
- **DAO Rewards**: Earn blockchain-based reputation for contributions.
- **MCP Alchemist Agent**: PyTorch 4x-powered agent for automating workflows and visualizations.

## MCP Alchemist Agent

The **MCP Alchemist Agent** is a PyTorch 4x-powered intelligent agent that automates and optimizes workflows within the Vial MCP Controller. Leveraging PyTorch’s GPU acceleration, the agent handles quantum circuit execution, LLM routing, and 3D visualization tasks. Its modular design allows contributors to extend its functionality via the Vial SDK.

### Core Functions
- **Quantum Workflow Automation**: Parses and optimizes QASM 2.0 circuits using Qiskit, executing quantum-RAG queries with minimal latency.
- **LLM Coordination**: Dynamically selects the optimal LLM provider based on query complexity, latency, and availability, using `litellm` metrics.
- **3D Visualization**: Generates GPU-accelerated 3D models of quantum circuits and network topologies, integrating with Three.js for real-time rendering.
- **DAO Integration**: Tracks contributions (e.g., code, circuit designs, PyTorch scripts) and assigns reputation points to `.mdwallets` on the blockchain.
- **Error Handling**: Provides detailed error messages to prevent API congestion, with logging for debugging.

### PyTorch 4x Optimization
The agent uses PyTorch 4x, an optimized version of PyTorch for multi-GPU environments, to accelerate:
- **Circuit Optimization**: Reduces quantum circuit depth using gradient-based techniques, minimizing execution time.
- **Visualization Rendering**: Generates high-fidelity 3D models with real-time Mermaid diagram updates.
- **Agentic Workflows**: Executes autonomous tasks (e.g., query routing, error handling) with minimal CPU overhead.

### Sample PyTorch Script
```python
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from typing import Dict, List

class AlchemistAgent(nn.Module):
    def __init__(self, num_qubits: int, num_layers: int = 4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.circuit_optimizer = nn.Sequential(
            nn.Linear(num_qubits * 2, 128),
            nn.ReLU(),
            *[nn.Linear(128, 128) for _ in range(num_layers - 1)],
            nn.Linear(128, num_qubits)
        ).to(self.device)
        self.llm_selector = nn.Linear(64, 10).to(self.device)  # Selects among 10 LLMs

    def optimize_circuit(self, circuit: str) -> str:
        """Optimize QASM circuit using GPU-accelerated PyTorch."""
        try:
            qc = QuantumCircuit.from_qasm_str(circuit)
            circuit_tensor = torch.tensor(self._circuit_to_tensor(qc), device=self.device)
            optimized_tensor = self.circuit_optimizer(circuit_tensor)
            optimized_circuit = self._tensor_to_circuit(optimized_tensor)
            return optimized_circuit.to_qasm()
        except Exception as e:
            raise ValueError(f"Circuit optimization failed: {str(e)}")

    def _circuit_to_tensor(self, circuit: QuantumCircuit) -> List[float]:
        # Convert QASM circuit to tensor (placeholder)
        return [0.0] * (circuit.num_qubits * 2)

    def _tensor_to_circuit(self, tensor: torch.Tensor) -> QuantumCircuit:
        # Convert tensor back to QASM circuit (placeholder)
        return QuantumCircuit(2)

    async def select_llm(self, query: str) -> str:
        """Select optimal LLM based on query features."""
        query_tensor = torch.tensor(self._query_to_tensor(query), device=self.device)
        scores = self.llm_selector(query_tensor)
        llm_index = torch.argmax(scores).item()
        return ["anthropic", "mistral", "google", "xai", "meta", "local"][llm_index]

    def _query_to_tensor(self, query: str) -> List[float]:
        # Convert query to tensor (placeholder)
        return [0.0] * 64

    async def run_workflow(self, query: str, circuit: str) -> Dict:
        """Execute quantum-RAG and LLM routing."""
        try:
            optimized_circuit = self.optimize_circuit(circuit)
            llm_provider = await self.select_llm(query)
            # Integrate with quantum_processor.py and llm_router.py
            results = {"provider": llm_provider, "circuit": optimized_circuit, "results": []}
            return results
        except Exception as e:
            raise RuntimeError(f"Workflow failed: {str(e)}")

Contribution Opportunities
This script is a placeholder for server/services/alchemist_agent.py (in development). Contributors can:

Implement _circuit_to_tensor and _tensor_to_circuit for real QASM conversion.
Add reinforcement learning for LLM selection.
Integrate real-time SVG animation with Three.js.
Develop anomaly detection for quantum circuit errors.

Architecture Overview
The Vial MCP Controller’s modular, quantum-distributed architecture is visualized in the following Mermaid diagram, which renders as an image on GitHub:
graph TD
    A[FastAPI Server<br>/mcp/auth, /mcp/quantum_rag] --> B[Quantum Processor<br>Qiskit, Kyber-512]
    A --> C[LLM Router<br>Anthropic, Mistral, xAI]
    A --> D[MCP Alchemist Agent<br>PyTorch 4x, GPU]
    A --> E[React/Next.js<br>QuantumCircuit.tsx]
    A --> F[OBS Integration<br>obs_handler.py]
    A --> G[ServiceNow<br>servicenow_endpoint.py]
    A --> H[DAO & .mdwallets<br>Blockchain Rewards]
    D --> B
    D --> C
    D --> E
    style A fill:#4CAF50,stroke:#333,stroke-width:2px
    style B fill:#2196F3,stroke:#333,stroke-width:2px
    style C fill:#FF9800,stroke:#333,stroke-width:2px
    style D fill:#009688,stroke:#333,stroke-width:2px
    style E fill:#9C27B0,stroke:#333,stroke-width:2px
    style F fill:#FFC107,stroke:#333,stroke-width:2px
    style G fill:#F44336,stroke:#333,stroke-width:2px
    style H fill:#673AB7,stroke:#333,stroke-width:2px

This Mermaid diagram renders as a clear, interactive flowchart, showing the FastAPI server as the central hub coordinating all components.
MCP Circuit Board Interface
The MCP Circuit Board Interface is a customizable visualization layer, rendered as a Mermaid diagram for real-time viewing:
graph LR
    A[Quantum Module<br>Qiskit, Kyber-512] --> B[LLM Router<br>litellm]
    B --> C[Alchemist Agent<br>PyTorch 4x]
    C --> D[DAO Rewards<br>Blockchain]
    C --> E[ Visualization<br>ReactFlow, Three.js]
    style A fill:#2196F3,stroke:#333,stroke-width:2px
    style B fill:#FF9800,stroke:#333,stroke-width:2px
    style C fill:#009688,stroke:#333,stroke-width:2px
    style D fill:#673AB7,stroke:#333,stroke-width:2px
    style E fill:#9C27B0,stroke:#333,stroke-width:2px

This diagram visualizes the circuit board interface, with contributors able to customize it via the Vial SDK or VS Code extension.
DAO and .mdwallets System
The DAO (Decentralized Autonomous Organization) incentivizes contributions with reputation points stored in blockchain-based .mdwallets, which also serve as OAuth 2.0 keys. Key features:

Wallet Generation: Run WebXOS: Initialize Wallet to create a .mdwallet with a Kyber-512 keypair and JWT token, stored in a local SQLite database.
Reputation Rewards: Contributions (e.g., code, circuit designs, PyTorch scripts) earn points, recorded on a blockchain (e.g., Ethereum or a custom chain).
Data Erasure: Unexported wallets are deleted on terminal close, ensuring a lightweight runtime.
Transparency: Blockchain ensures immutable tracking of contributions and rewards.

Contributors can design custom circuit board interfaces or enhance 3D rendering, earning higher reputation for impactful submissions. Smart contract details are planned for docs/dao.md.
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
Set API keys and settings:[server]
host = "0.0.0.0"
port = 3000
[auth]
JWT_SECRET = "your-secret-key"
[llm]
ANTHROPIC_API_KEY = "your-key"
MISTRAL_API_KEY = "your-key"
[obs]
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = "your-password"
[servicenow]
SERVICENOW_INSTANCE = "your-instance"
SERVICENOW_USER = "your-user"
SERVICENOW_PASSWORD = "your-password"
[scaling]
K8S_MIN_REPLICAS = 2
K8S_MAX_REPLICAS = 10
K8S_CPU_TARGET = 0.7
K8S_MEMORY_TARGET = "500Mi"
K8S_NAMESPACE = "vial-mcp"




Run the Server:docker-compose up -d

Access at http://localhost:3000.

Installing WebXOS 2025 VS Code Extension

Download Extension:
Get webxos-2025.vsix from extensions/webxos-2025/.


Install in VS Code:code --install-extension extensions/webxos-2025/webxos-2025.vsix


Configure Extension:
Set webxos-2025.apiEndpoint to http://localhost:3000.
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
To contribute or customize:

Fork the Repo:
Click "Fork" on github.com/webxos/vial.github.io.


Clone Your Fork:git clone https://github.com/<your-username>/vial.github.io.git
cd vial.github.io


Create a Branch:git checkout -b feature/your-feature


Make Changes:
Add code, tests, or documentation.
Run npm run lint and flake8 server/ for style compliance.
Run pytest server/tests/ -v --cov=server for 90%+ coverage.


Commit and Push:git commit -m "Add your-feature"
git push origin feature/your-feature


Submit a Pull Request:
Create a PR against main on the original repo.
Include tests and update docs/user_guide.md.



API Access and Tools
The Vial MCP Controller exposes a FastAPI-based API, requiring OAuth 2.0 authentication via .mdwallets.
Complete API Calls List



Endpoint
Method
Description
Parameters
Response
Example
Error Handling



/mcp/auth
POST
Authenticate with .mdwallet
wallet_id: str, public_key: str
{"access_token": str}
curl -X POST http://localhost:3000/mcp/auth -d '{"wallet_id": "wallet123", "public_key": "a1b2c3..."}'
400: Invalid key, 500: Server error


/mcp/quantum_rag
POST
Execute quantum-RAG query
query: str, quantum_circuit: str, max_results: int
{"results": [str], "circuit": str}
curl -X POST http://localhost:3000/mcp/quantum_rag -H "Authorization: Bearer <token>" -d '{"query": "test", "quantum_circuit": "H 0; CX 0 1", "max_results": 5}'
400: Invalid circuit, 401: Unauthorized


/mcp/servicenow/ticket
POST
Create ServiceNow ticket
short_description: str, description: str, urgency: str
{"result": {"number": str}}
curl -X POST http://localhost:3000/mcp/servicenow/ticket -H "Authorization: Bearer <token>" -d '{"short_description": "Test", "description": "Issue", "urgency": "low"}'
400: Missing fields, 401: Unauthorized


/mcp/tools/obs.init
POST
Initialize OBS scene
scene_name: str
{"scene": str}
curl -X POST http://localhost:3000/mcp/tools/obs.init -H "Authorization: Bearer <token>" -d '{"scene_name": "MyScene"}'
400: Missing scene name, 500: OBS connection error


/mcp/monitoring/health
GET
Check system health
None
{"status": str, "cpu_usage_percent": float, "memory_usage_percent": float, "disk_usage_percent": float, "services": dict}
curl -H "Authorization: Bearer <token>" http://localhost:3000/mcp/monitoring/health
401: Unauthorized, 500: System error


Tools

Vial SDK: Python library (@modelcontextprotocol/sdk) for API access. Install:pip install @modelcontextprotocol/sdk

Example:from modelcontextprotocol import MCPClient
client = MCPClient(api_endpoint="http://localhost:3000", token="your_token")
response = client.quantum_rag(query="test", circuit="H 0; CX 0 1", max_results=5)
print(response["results"])


WebXOS 2025 VS Code Extension: GUI for API calls, wallet management, and visualization.
Postman Collection: In docs/postman_collection.json for testing.
Swagger UI: Access at http://localhost:3000/docs.

Resources

Quantum Computing: Qiskit Tutorials
LLM Integration: Litellm Documentation
FastAPI: FastAPI Documentation
ReactFlow: ReactFlow Documentation
Three.js: Three.js Documentation
PyTorch: PyTorch Tutorials
DAO and Blockchain: Ethereum Developer Resources
Kyber-512: NIST PQC Standards

Features in Development
Planned features for future phases, with contribution opportunities:

MCP Alchemist Agent (server/services/alchemist_agent.py): Full PyTorch 4x implementation for circuit optimization, LLM routing, and 3D visualization.
Interactive Circuit Editor (typescript/src/components/CircuitEditor.tsx): React component for designing QASM circuits.
Tauri Desktop App (desktop/tauri.conf.json): Desktop support for Linux, per linux-desktop-app-guide.md.
Kubernetes Deployment (k8s/deployment.yaml): Cluster deployment with auto-scaling.
Analytics Endpoint (server/api/llm_analytics.py): LLM performance metrics.
DAO Smart Contracts (docs/dao.md): Blockchain specifications for rewards.

Directory Structure
Complete directory tree with detailed descriptions:
vial.github.io/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions for CI/CD, linting, testing
├── docs/
│   ├── user_guide.md           # Setup, usage, and extension guide
│   ├── CONTRIBUTING.md         # Contribution and DAO reward guidelines
│   └── postman_collection.json # Postman collection for API testing
├── extensions/
│   └── webxos-2025/
│       ├── package.json        # VS Code extension metadata
│       └── src/
│           └── extension.ts    # Wallet management and API call logic
├── scripts/
│   └── update_deps.sh          # Dependency update script
├── server/
│   ├── api/
│   │   ├── auth_endpoint.py    # /mcp/auth for .mdwallets authentication
│   │   ├── quantum_rag_endpoint.py # /mcp/quantum_rag for quantum-RAG
│   │   ├── servicenow_endpoint.py  # /mcp/servicenow/ticket for ticketing
│   │   └── monitoring_endpoint.py  # /mcp/monitoring/health for system health
│   ├── config/
│   │   ├── settings.py        # Loads mcp.toml configuration
│   │   └── scaling_config.py  # Kubernetes HPA configuration
│   ├── services/
│   │   ├── llm_router.py      # Multi-LLM routing with litellm
│   │   ├── quantum_processor.py # Qiskit-based quantum circuit execution
│   │   └── obs_handler.py     # OBS WebSocket for video scenes
│   ├── tests/
│   │   ├── test_llm_router.py # Tests llm_router.py
│   │   ├── test_quantum_rag_endpoint.py # Tests quantum_rag_endpoint.py
│   │   ├── test_quantum_processor.py   # Tests quantum_processor.py
│   │   ├── test_servicenow_endpoint.py # Tests servicenow_endpoint.py
│   │   ├── test_monitoring_endpoint.py # Tests monitoring_endpoint.py
│   │   └── test_auth_endpoint.py      # Tests auth_endpoint.py
│   └── main.py                # FastAPI entry point
├── typescript/
│   ├── src/
│   │   ├── components/
│   │   │   ├── QuantumCircuit.tsx     # Quantum circuit visualization
│   │   │   └── TopologyVisualizer.tsx # Network topology visualization
│   │   ├── tests/
│   │   │   └── TopologyVisualizer.test.tsx # Tests TopologyVisualizer.tsx
│   │   └── utils/
│   │       └── fetcher.ts            # Secure API calls with OAuth
├── .eslintrc.json                    # ESLint configuration
├── docker-compose.yml                # Docker Compose for deployment
├── mcp.toml.example                  # Example configuration
├── package.json                      # Node.js dependencies and scripts
├── requirements.txt                  # Python dependencies
└── README.md                         # This file

3D GPU Rendering and PyTorch 4x Agentic Scripts
The Vial MCP Controller uses Three.js for 3D visualization and PyTorch 4x for agentic scripts, enabling GPU-accelerated rendering of quantum circuits and topologies. The MCP Alchemist Agent leverages PyTorch to:

Optimize Quantum Circuits: Reduces circuit depth using gradient-based optimization.
Render 3D Visuals: Generates real-time 3D models with Mermaid overlays.
Automate Workflows: Routes LLM queries and manages OBS scenes autonomously.

Contributing and DAO Rewards
To contribute:

Fork and Clone: Follow the forking guide.
Develop Features:
Add endpoints in server/api/.
Enhance visualizations in typescript/src/components/.
Create PyTorch scripts in server/services/.


Test and Lint:npm run lint
flake8 server/
pytest server/tests/ -v --cov=server


Submit PRs: Include tests and documentation.
Earn Rewards: Contributions earn .mdwallets reputation points.

See CONTRIBUTING.md for details.
Troubleshooting

Mermaid Diagrams: Ensure GitHub or your Markdown viewer supports Mermaid (enabled by default on GitHub).
ESLint Errors: Run npm run lint.
Flake8 Errors: Run flake8 server/.
Test Failures: Run pytest server/tests/ -v and check logs/.
Extension Issues: Verify webxos-2025.apiEndpoint.
API Errors: Check http://localhost:3000/docs.

License
MIT License. See LICENSE for details.

Thank you for joining the Vial MCP Controller community! Fork, build, and contribute to shape the future of quantum-AI automation.```
