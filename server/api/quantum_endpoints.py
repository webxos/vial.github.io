from fastapi import APIRouter, Depends
from server.quantum.quantum_sync import QuantumVisualSync
from server.services.advanced_logging import AdvancedLogger
from pydantic import BaseModel


class QuantumRequest(BaseModel):
    vial_id: str


router = APIRouter()
logger = AdvancedLogger()


@router.post("/sync")
async def sync_quantum(request: QuantumRequest):
    quantum_sync = QuantumVisualSync(request.vial_id)
    result = quantum_sync.sync_quantum_state(request.vial_id)
    logger.log("Quantum sync executed", extra={"vial_id": request.vial_id})
    return result
