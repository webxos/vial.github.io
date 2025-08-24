```python
import qiskit
from fastapi import HTTPException

class QuantumSimulator:
    def __init__(self):
        self.backend = qiskit.Aer.get_backend('qasm_simulator')

    def simulate_quantum_circuit(self, n_qubits: int) -> dict:
        """Simulate a quantum circuit.
        
        Parameters:
            n_qubits: Number of qubits
        
        Returns:
            Simulation result dictionary.
        """
        try:
            circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
            circuit.h(range(n_qubits))
            circuit.measure_all()
            job = qiskit.execute(circuit, self.backend, shots=1024)
            result = job.result().get_counts()
            return {"n_qubits": n_qubits, "result": result, "timestamp": "06:48 PM EDT"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

simulator = QuantumSimulator()

if __name__ == "__main__":
    print(simulator.simulate_quantum_circuit(2))
```
