import pytest
from server.services.mcp_alchemist import MCPAlchemist
from server.logging_config import logger
import uuid

@pytest.fixture
def alchemist():
    return MCPAlchemist()


@pytest.mark.asyncio
async def test_quantum_circuit(alchemist):
    request_id = str(uuid.uuid4())
    try:
        
        params = {"vial_id": "vial1", "qubits": 2}
        result = await alchemist.execute_quantum_circuit(params, request_id)
        assert result["status"] == "executed"
        assert "circuit_id" in result
        logger.info(f"Quantum circuit test passed", request_id=request_id)
    except Exception as e:
        logger.error(f"Quantum circuit test failed: {str(e)}", request_id=request_id)
        raise


@pytest.mark.asyncio
async def test_train_model(alchemist):
    request_id = str(uuid.uuid4())
    try:
        
        params = {"vial_id": "vial2", "epochs": 10}
        result = await alchemist.train_model(params, request_id)
        assert result["status"] == "trained"
        assert "accuracy" in result
        logger.info(f"Train model test passed", request_id=request_id)
    except Exception as e:
        logger.error(f"Train model test failed: {str(e)}", request_id=request_id)
        raise
