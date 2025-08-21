from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer
from server.quantum.quantum_sync import execute_quantum_circuit

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


@router.post("/execute")
async def execute_circuit(circuit: str, token: str = Depends(oauth2_scheme)):
    return await execute_quantum_circuit(circuit)
