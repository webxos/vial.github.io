from fastapi import APIRouter, Depends
from prometheus_client import Counter, Gauge
from server.security.oauth2 import validate_token
import grpc
import time

router = APIRouter(prefix="/api/lattice")

LATTICE_API_CALLS = Counter(
    'mcp_lattice_api_calls_total',
    'Total Lattice MCP API calls',
    ['method']
)

LATTICE_RESPONSE_TIME = Gauge(
    'mcp_lattice_response_time_seconds',
    'Lattice MCP response time'
)

class LatticeMCPWrapper:
    def __init__(self):
        self.channel = grpc.insecure_channel(os.getenv("LATTICE_ENDPOINT", "localhost:50051"))

    async def call_api(self, method: str, token: str = Depends(validate_token)):
        start_time = time.time()
        LATTICE_API_CALLS.labels(method=method).inc()
        # Simulate Lattice API call (replace with actual gRPC stub)
        response = {"data": f"{method} response"}
        latency = time.time() - start_time
        LATTICE_RESPONSE_TIME.set(latency)
        return response
