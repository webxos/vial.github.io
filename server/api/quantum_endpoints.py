from fastapi import APIRouter, Depends, HTTPException
from server.services.mcp_alchemist import MCPAlchemist
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1/quantum", tags=["quantum"])


@router.post("/execute_circuit")
async def execute_quantum_circuit(params: dict, alchemist: MCPAlchemist = Depends()):
    request_id = str(uuid.uuid4())
    try:
        result = await alchemist.execute_quantum_circuit(params, request_id)
        logger.info(f"Executed quantum circuit for vial {params.get('vial_id')}", request_id=request_id)
        return {"result": result, "request_id": request_id}
    except Exception as e:
        logger.error(f"Quantum circuit error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuit_status/{circuit_id}")
async def get_circuit_status(circuit_id: str, alchemist: MCPAlchemist = Depends()):
    request_id = str(uuid.uuid4())
    try:
        status = await alchemist.get_circuit_status(circuit_id, request_id)
        logger.info(f"Retrieved status for circuit {circuit_id}", request_id=request_id)
        return {"status": status, "request_id": request_id}
    except Exception as e:
        logger.error(f"Circuit status error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
