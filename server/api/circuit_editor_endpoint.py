from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.services.quantum_circuit_optimizer import optimize_circuit
from server.config.database import get_db
from server.models.dao_repository import DAORepository
from sqlalchemy.orm import Session
from fastapi.security import OAuth2AuthorizationCodeBearer
from qiskit import QuantumCircuit
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO, filename="logs/circuit_editor.log")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/mcp/auth/authorize",
    tokenUrl="/mcp/auth/token",
    scheme_name="OAuth2PKCE"
)

class CircuitRequest(BaseModel):
    circuit_qasm: str
    wallet_id: str
    optimize: bool = True

@router.post("/design")
async def design_circuit(request: CircuitRequest, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        # Validate QASM
        try:
            qc = QuantumCircuit.from_qasm_str(request.circuit_qasm)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid QASM: {str(e)}")
        
        # Optimize circuit if requested
        optimized_qasm = request.circuit_qasm
        if request.optimize:
            optimized_qasm = await optimize_circuit(request.circuit_qasm)
        
        # Save to database and award DAO points
        dao_repo = DAORepository(db)
        dao_repo.add_reputation(request.wallet_id, points=10)  # Award 10 points for circuit design
        
        return {"circuit": optimized_qasm, "reputation_points": dao_repo.get_reputation(request.wallet_id)}
    except Exception as e:
        logging.error(f"Circuit design error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
