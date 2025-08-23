from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.mcp.auth import oauth2_scheme
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid

router = APIRouter()

class QuantumCircuitRequest(BaseModel):
    qubits: int
    gates: list[str]

@router.post("/quantum/circuit")
async def build_quantum_circuit(request: QuantumCircuitRequest, token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(request.qubits)
        for gate in request.gates:
            if gate == "h":
                circuit.h(range(request.qubits))
            elif gate == "cx":
                circuit.cx(0, 1)
        svg = circuit.draw(output="svg")
        logger.log(f"Quantum circuit built with {request.qubits} qubits", request_id=request_id)
        return {"circuit": svg, "request_id": request_id}
    except Exception as e:
        logger.log(f"Quantum circuit error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
