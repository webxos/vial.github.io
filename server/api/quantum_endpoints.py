from fastapi import APIRouter
from server.quantum.quantum_sync import QuantumSync

router = APIRouter()


@router.post("/execute")
async def execute_quantum_circuit(circuit_data: dict):
    quantum_sync = QuantumSync(None)
    return quantum_sync.sync_quantum_state(circuit_data.get("vial_id"))
