from mcp.transport import TransportLayer
from fastapi import WebSocket
from prometheus_client import Counter
from server.security.quantum_crypto import QuantumKeyGenerator
import asyncio
import json

TRANSPORT_MESSAGES = Counter(
    'mcp_transport_messages_total',
    'Total messages processed by transport layer',
    ['direction']
)

class MCPTransportMetrics(TransportLayer):
    def __init__(self):
        super().__init__()
        self.keygen = QuantumKeyGenerator()
        self.connections = {}
    
    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket connections with metrics."""
        await websocket.accept()
        self.connections[client_id] = websocket
        TRANSPORT_MESSAGES.labels(direction="inbound").inc()
        try:
            while True:
                data = await websocket.receive_json()
                encrypted_data = self.keygen.encrypt_data(json.dumps(data))
                await self.broadcast(encrypted_data, client_id)
                TRANSPORT_MESSAGES.labels(direction="outbound").inc()
        except Exception as e:
            await websocket.close()
            del self.connections[client_id]
    
    async def broadcast(self, data: str, sender_id: str):
        """Broadcast encrypted data with metrics."""
        for client_id, ws in self.connections.items():
            if client_id != sender_id:
                try:
                    await ws.send_json({"data": data, "sender": sender_id})
                    TRANSPORT_MESSAGES.labels(direction="broadcast").inc()
                except Exception:
                    del self.connections[client_id]
