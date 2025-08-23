import os
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from qiskit import QuantumCircuit, Aer, execute
from qiskit_aer import AerSimulator
from mcp import tool
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class QuantumTask(BaseModel):
    qubits: int = 2
    depth: int = 1
    shots: int = 1024

class QuantumResult(BaseModel):
    counts: Dict[str, int]
    signature: Optional[str] = None

def sign_result(counts: Dict[str, int], private_key: str) -> str:
    """Sign quantum result for wallet .md export."""
    private_key_bytes = bytes.fromhex(private_key)
    key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    data = str(counts).encode()
    return key.sign(data).hex()

@tool("quantum_sync")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def run_quantum_task(task: QuantumTask) -> QuantumResult:
    """Run a quantum circuit and return signed results."""
    try:
        circuit = QuantumCircuit(task.qubits, task.qubits)
        circuit.h(range(task.qubits))  # Apply Hadamard gates
        circuit.measure_all()
        
        simulator = AerSimulator()
        job = execute(circuit, simulator, shots=task.shots)
        result = job.result()
        counts = result.get_counts()
        
        private_key = os.getenv("WALLET_ENCRYPTION_KEY", "")
        signature = sign_result(counts, private_key) if private_key else None
        
        return QuantumResult(counts=counts, signature=signature)
    except Exception as e:
        logger.error(f"Quantum task failed: {str(e)}")
        raise
