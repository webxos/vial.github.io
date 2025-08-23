from typing import Dict, Any
from qiskit import QuantumCircuit
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

class QuantumTasks:
    def __init__(self):
        self.memory_manager = MemoryManager()

    async def build_quantum_circuit(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        try:
            qubits = params.get("qubits", 4)
            gates = params.get("quantum_logic", {}).get("gates", ["H", "CNOT"])
            circuit = QuantumCircuit(qubits)
            for gate in gates:
                if gate == "H":
                    circuit.h(range(qubits))
                elif gate == "CNOT":
                    for i in range(qubits - 1):
                        circuit.cx(i, i + 1)
            circuit_data = {"qubits": qubits, "gates": gates, "circuit": str(circuit)}
            await self.memory_manager.save_task_relationship(
                "quantum_circuit",
                {"quantum_logic": circuit_data, "training_data": {}, "related_tasks": ["quantum_circuit"]},
                request_id
            )
            logger.info(f"Built quantum circuit with {qubits} qubits", request_id=request_id)
            return {"status": "circuit_built", "circuit": circuit_data, "request_id": request_id}
        except Exception as e:
            logger.error(f"Quantum circuit error: {str(e)}", request_id=request_id)
            raise
