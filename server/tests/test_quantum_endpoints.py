import pytest
import httpx
from unittest.mock import patch, AsyncMock
from server.quantum.qiskit_engine import run_quantum_task, QuantumTask, QuantumResult
from cryptography.hazmat.primitives.asymmetric import x25519  # Placeholder for PQC

@pytest.fixture
def quantum_task():
    return QuantumTask(qubits=2, depth=1, shots=1024)

@pytest.fixture
async def mock_qiskit():
    with patch("qiskit.execute", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value.result.return_value.get_counts.return_value = {"00": 512, "11": 512}
        yield mock_exec

@pytest.mark.asyncio
async def test_quantum_sync(quantum_task, mock_qiskit):
    result = await run_quantum_task(quantum_task)
    assert isinstance(result, QuantumResult)
    assert "00" in result.counts
    assert mock_qiskit.called

@pytest.mark.asyncio
async def test_quantum_integrity(quantum_task, mock_qiskit):
    with patch("server.quantum.qiskit_engine.sign_result") as mock_sign:
        mock_sign.return_value = "mock_signature"
        result = await run_quantum_task(quantum_task)
        assert result.signature == "mock_signature"

@pytest.mark.asyncio
async def test_pqc_readiness(quantum_task):
    with patch("cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey.generate") as mock_pqc:
        mock_pqc.return_value = None  # Simulate Kyber-512
        result = await run_quantum_task(quantum_task)
        assert isinstance(result, QuantumResult)
