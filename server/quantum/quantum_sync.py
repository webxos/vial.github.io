from qiskit import QuantumCircuit, Aer, execute
from server.services.vial_manager import VialManager
from server.models.visual_components import ComponentModel
from server.logging import logger
import hashlib


class QuantumVisualSync:
    def __init__(self, vial_manager: VialManager = None):
        self.vial_manager = vial_manager or VialManager()
        self.backend = Aer.get_backend('qasm_simulator')

    def create_quantum_circuit_from_visual(self, components: list[ComponentModel]):
        try:
            qc = QuantumCircuit(4, 4)
            for component in components:
                if component.type == "agent":
                    vial_id = component.config.get("vial_id", "default")
                    if vial_id in self.vial_manager.agents:
                        qc.h(hash(vial_id) % 4)
            result = execute(qc, self.backend, shots=1).result()
            quantum_hash = hashlib.sha256(str(result).encode()).hexdigest()[:64]
            logger.log(f"Quantum circuit created for components: {quantum_hash}")
            return {"quantum_hash": quantum_hash}
        except Exception as e:
            logger.log(f"Quantum circuit error: {str(e)}")
            return {"error": str(e)}

    def sync_quantum_state(self, vial_id: str):
        try:
            if vial_id not in self.vial_manager.agents:
                raise ValueError(f"Vial {vial_id} not found")
            qc = QuantumCircuit(4, 4)
            qc.h(hash(vial_id) % 4)
            result = execute(qc, self.backend, shots=1).result()
            quantum_state = result.get_counts()
            logger.log(f"Quantum state synced for vial: {vial_id}")
            return {"quantum_state": quantum_state}
        except Exception as e:
            logger.log(f"Quantum sync error: {str(e)}")
            return {"error": str(e)}
