from fastapi import APIRouter, Depends
from server.quantum.quantum_sync import quantum_sync
from server.security import verify_token

router = APIRouter()


async def execute_quantum_circuit(
    circuit_data: dict,
    token: str = Depends(verify_token)
):
    result = quantum_sync.execute_circuit(circuit_data)
    return result


async def get_quantum_status(token: str = Depends(verify_token)):
    status = quantum_sync.get_status()
    return status
