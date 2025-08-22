from fastapi import APIRouter
from server.quantum.quantum_sync import QuantumSync

router = APIRouter()

quantum_sync = QuantumSync(None)

@router.post("/sync")
async def sync_quantum(vial_id: str):
    return quantum_sync.sync_quantum_state(vial_id)
