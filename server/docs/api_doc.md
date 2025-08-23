Vial MCP Controller API Documentation
Base URL
http://localhost:8000/v1
Frontend
GET /
Renders the main interface with wallet and agent visualization.

Response: HTML page with console, balance, buttons, and Three.js canvas.
Features:
Displays $WEBXOS Balance and Reputation.
Buttons for API Access, Authenticate, Void, Troubleshoot, Quantum Link, Export, Import.
Three.js canvas for 3D vial visualization.


Dependencies: Requires /_next/static/chunks/main.js and /js/threejs_integrations.js.

Authentication
POST /v1/auth/token
Generate a JWT token.

Request Body: { "network_id": str, "session_id": str }
Response: { "token": str, "request_id": str }

GET /v1/auth/validate
Validate a JWT token.

Response: { "status": str, "request_id": str }

JSON-RPC
POST /v1/jsonrpc
Execute tasks via JSON-RPC 2.0.

Request Body: { "jsonrpc": "2.0", "method": str, "params": dict, "id": str }
Methods:
vial_train: Train a vial (params: { "vial_id": str, "network_id": str })
agent_coord: Coordinate 4x vial agents (params: { "network_id": str })
quantum_circuit: Build quantum circuit (params: { "qubits": int })


Response: { "jsonrpc": "2.0", "result": dict, "id": str, "request_id": str }

Wallet Operations
POST /v1/upload/wallet
Upload a .md wallet file.

Request Body: Form-data with file (.md file)
Response: { "status": str, "network_id": str, "request_id": str }

POST /v1/wallet/import
Import a .md wallet.

Request Body: { "file": str }
Response: { "network_id": str, "balance": float, "request_id": str }

POST /v1/wallet/export
Export a wallet as .md.

Request Body: { "network_id": str }
Response: { "markdown": str, "request_id": str }

Monitoring
GET /v1/monitoring/health
Check system health.

Response: { "status": str, "db": bool, "agents": dict, "wallet": bool, "response_time": float, "request_id": str }

GET /v1/monitoring/logs
Retrieve error logs.

Response: { "logs": str, "request_id": str }

Troubleshooting
GET /v1/troubleshoot/status
Check system status.

Response: { "status": str, "db": bool, "agents": dict, "wallet": bool, "request_id": str }

WebSocket
WS /v1/mcp/ws
Real-time task execution.

Message Format: { "task": str, "params": dict }
Response Format: { "result": dict, "request_id": str, "session_id": str }
