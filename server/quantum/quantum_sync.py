from server.logging import logger
import qiskit


class QuantumSync:
    def __init__(self):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def execute_circuit(self, circuit_data: dict):
        try:
            circuit = circuit_data.get("circuit")
            if not circuit:
                raise ValueError("Invalid circuit")
            result = qiskit.execute(circuit, self.backend).result()
            return {"status": "success", "counts": result.get_counts()}
        except Exception as e:
            logger.error(f"Quantum circuit execution failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def get_status(self):
        return {"status": "running", "backend": self.backend.name()}


quantum_sync = QuantumSync()
