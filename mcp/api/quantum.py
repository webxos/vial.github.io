from fastapi import APIRouter, Depends
from ..quantum.entanglement import entanglement_manager
from ..auth import get_current_user

router = APIRouter()

@router.post("/entangle")
async def entangle_qubits(link_id: str, qubit_count: int, current_user: dict = Depends(get_current_user)):
    return entanglement_manager.entangle_qubits(link_id, qubit_count)
