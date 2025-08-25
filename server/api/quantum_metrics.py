from fastapi import APIRouter, Depends, Response
from prometheus_client import generate_latest, Counter, Histogram
from server.security.oauth2 import validate_token
from server.services.quantum_topology import QuantumTopologyService

router = APIRouter(prefix="/metrics/quantum")
quantum_service = QuantumTopologyService()

QUANTUM_CIRCUIT_EXECUTIONS = Counter(
    'mcp_quantum_circuit_executions_total',
    'Total quantum circuit executions',
    ['circuit_type', 'qubits_count']
)

QUANTUM_LATENCY = Histogram(
    'mcp_quantum_execution_duration_seconds',
    'Quantum execution latency',
    ['circuit_type']
)

@router.get("")
async def quantum_metrics(token: str = Depends(validate_token)):
    """Expose quantum agent metrics."""
    # Simulate metric updates (replace with actual data in production)
    QUANTUM_CIRCUIT_EXECUTIONS.labels(circuit_type="h-gate", qubits_count=8).inc()
    QUANTUM_LATENCY.labels(circuit_type="h-gate").observe(0.5)
    
    return Response(generate_latest(), media_type="text/plain")
