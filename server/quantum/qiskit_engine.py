from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from server.models.visual_components import ComponentModel
from server.logging import logger
import hashlib


class QiskitEngine:
    def __init__(self):
        self.backend = AerSimulator()

    def build_circuit_from_components(self, components: list[ComponentModel]) -> dict:
        try:
            qc = QuantumCircuit(4, 4)
            for component in components:
                if component.type == "agent":
                    vial_id = component.config.get("vial_id", "default")
                    qc.h(hash(vial_id) % 4)
                elif component.type == "api_endpoint":
                    qc.x(hash(component.id) % 4)
            transpiled_qc = transpile(qc, self.backend, optimization_level=1)
            qasm = transpiled_qc.qasm()
            quantum_hash = hashlib.sha256(qasm.encode()).hexdigest()[:64]
            logger.log(f"Built quantum circuit: {quantum_hash}")
            return {"qasm": qasm, "quantum_hash": quantum_hash}
        except Exception as e:
            logger.log(f"Circuit build error: {str(e)}")
            return {"error": str(e)}

    def run_circuit(self, qasm: str) -> dict:
        try:
            qc = QuantumCircuit.from_qasm_str(qasm)
            result = self.backend.run(qc, shots=1).result()
            counts = result.get_counts()
            logger.log(f"Circuit executed: {counts}")
            return {"counts": counts}
        except Exception as e:
            logger.log(f"Circuit run error: {str(e)}")
            return {"error": str(e)}
