from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qiskit import QuantumCircuit, Aer, execute


router = APIRouter()


class QuantumRequest(BaseModel):
    circuit: dict
    backend: str = "qasm_simulator"


@router.post("/execute")
async def execute_circuit(request: QuantumRequest):
    try:
        circuit = QuantumCircuit.from_dict(request.circuit)
        backend = Aer.get_backend(request.backend)
        result = execute(circuit, backend, shots=1024).result()
        counts = result.get_counts()
        return {"status": "success", "counts": counts}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
