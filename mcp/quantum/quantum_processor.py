import torch
from .quantum_state import quantum_state

class QuantumProcessor:
    def __init__(self):
        self.device = torch.device("cpu")

    def process_qubits(self, qubits):
        tensor = torch.tensor(qubits, device=self.device)
        return torch.sigmoid(tensor).tolist()

quantum_processor = QuantumProcessor()

def process_quantum_state(qubit_count: int):
    state = quantum_state.initialize_qubits(qubit_count)
    processed = quantum_processor.process_qubits(state["qubits"])
    return {"processed_qubits": processed, "time": "06:11 PM EDT, Aug 20, 2025"}
