```python
import pytest
from unittest.mock import patch, AsyncMock
from server.services.quantum_processor import QuantumProcessor
from fastapi import HTTPException

@pytest.mark.asyncio
async def test_quantum_processor_success():
    """Test successful quantum RAG execution."""
    processor = QuantumProcessor()
    with patch("qiskit_aer.AerSimulator.run", new=AsyncMock(return_value={"get_counts": lambda: {"00": 512, "11": 512}})):
        results = await processor.execute_quantum_rag(
            query="test query",
            circuit="OPENQASM 2.0; include 'qelib1.inc'; qreg q[2]; h q[0]; cx q[0],q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];",
            max_results=2
        )
        assert len(results) == 2
        assert results[0].startswith("Result-0 for query: test query")

@pytest.mark.asyncio
async def test_quantum_processor_invalid_circuit():
    """Test invalid quantum circuit handling."""
    processor = QuantumProcessor()
    with patch("qiskit.QuantumCircuit.from_qasm_str", side_effect=Exception("Invalid QASM")):
        with pytest.raises(HTTPException) as exc:
            await processor.execute_quantum_rag(
                query="test query",
                circuit="invalid qasm",
                max_results=2
            )
        assert exc.value.status_code == 400
        assert "Invalid quantum circuit" in str(exc.value.detail)

@pytest.mark.asyncio
async def test_quantum_processor_failure():
    """Test quantum processor failure."""
    processor = QuantumProcessor()
    with patch("qiskit_aer.AerSimulator.run", side_effect=Exception("Simulator error")):
        with pytest.raises(HTTPException) as exc:
            await processor.execute_quantum_rag(
                query="test query",
                circuit="OPENQASM 2.0; include 'qelib1.inc'; qreg q[2]; h q[0]; cx q[0],q[1];",
                max_results=2
            )
        assert exc.value.status_code == 500
        assert "Simulator error" in str(exc.value.detail)
```
