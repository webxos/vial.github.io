from mcp.transport import TransportLayer
from fastapi import WebSocket
from prometheus_client import Counter
from server.security.quantum_crypto import QuantumKeyGenerator
import asyncio
import logging
import json

TRANSPORT_MESSAGES = Counter(
    'mcp_transport_messages_total',
    'Total messages processed by transport layer',
    ['direction']
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPTransportLogging(TransportLayer):
    def __init__(self):
        super().__init__()
        self.keygen = QuantumKeyGenerator()
        self.connections = {}
    
    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket connections with logging and metrics."""
        await websocket.accept()
        self.connections[client_id] = websocket
        TRANSPORT_MESSAGES.labels(direction="inbound").inc()
        logger.info(f"New connection established: {client_id}")
        try:
            while True:
                data = await websocket.receive_json()
                logger.debug(f"Received data from {client_id}: {data}")
                encrypted_data = self.keygen.encrypt_data(json.dumps(data))
                await self.broadcast(encrypted_data, client_id)
                TRANSPORT_MESSAGES.labels(direction="outbound").inc()
                logger.info(f"Broadcast from {client_id} to {len(self.connections)-1} clients")
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
            await websocket.close()
            del self.connections[client_id]
    
    async def broadcast(self, data: str, sender_id: str):
        """Broadcast encrypted data with logging."""
        for client_id, ws in self.connections.items():
            if client_id != sender_id:
                try:
                    await ws.send_json({"data": data, "sender": sender_id})
                    TRANSPORT_MESSAGES.labels(direction="broadcast").inc()
                    logger.debug(f"Sent to {client_id}")
                except Exception:
                    logger.error(f"Failed to send to {client_id}")
                    del self.connections[client_id]
