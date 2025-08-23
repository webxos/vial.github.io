from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
from qiskit import QuantumCircuit
import uuid
import json
import os


router = APIRouter()


class QuantumCircuitRequest(BaseModel):
    components: list[dict]
    num_qubits: int = 2
    gate_types: list[str] = ["h", "cx"]


@router.post("/quantum/circuit/build")
async def build_quantum_circuit(
    request: QuantumCircuitRequest,
    token: str = Depends(oauth2_scheme)
):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        alchemist = Alchemist()
        circuit = QuantumCircuit(request.num_qubits)
        for comp in request.components:
            gate_type = comp.get("gate")
            qubits = comp.get("qubits", [])
            if gate_type not in request.gate_types:
                raise HTTPException(status_code=400, detail=f"Invalid gate: {gate_type}")
            if gate_type == "h":
                circuit.h(qubits[0])
            elif gate_type == "cx":
                circuit.cx(qubits[0], qubits[1])
        circuit_data = circuit.qasm()
        circuit_path = f"resources/quantum/circuit_{uuid.uuid4()}.qasm"
        os.makedirs(os.path.dirname(circuit_path), exist_ok=True)
        with open(circuit_path, "w") as f:
            f.write(circuit_data)
        logger.log(f"Quantum circuit built: {circuit_path}", request_id=request_id)
        return {"status": "built", "circuit_path": circuit_path}
    except Exception as e:
        logger.log(f"Quantum circuit error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
