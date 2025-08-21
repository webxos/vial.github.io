from fastapi import APIRouter, Depends
from server.models.mcp_alchemist import MCPAlchemist
from server.quantum.quantum_sync import QuantumRequest
from server.security import verify_jwt


router = APIRouter()


@router.post("/predict")
async def predict_quantum(
    request: QuantumRequest,
    token: str = Depends(verify_jwt)
):
    alchemist = MCPAlchemist()
    result = await alchemist.predict_quantum_outcome(request.circuit)
    return result
