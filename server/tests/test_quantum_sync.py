from server.quantum.quantum_sync import quantum_sync
from unittest.mock import patch


def test_quantum_sync_initialization():
    assert quantum_sync is not None

def test_quantum_sync_execute():
    with patch("qiskit.execute") as mock_execute:
        mock_execute.return_value.result.return_value.get_counts.return_value = {
            "00": 100
        }
        result = quantum_sync.execute_circuit({"circuit": "simple"})
        assert result["status"] == "success"
        assert result["counts"] == {"00": 100}

def test_quantum_sync_invalid_circuit():
    result = quantum_sync.execute_circuit({"circuit": "invalid"})
    assert result["status"] == "failed"
    assert "Invalid circuit" in result["error"]
