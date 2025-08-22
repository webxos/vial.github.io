from qiskit import QuantumCircuit, Aer, execute
from pydantic import BaseModel


class Component(BaseModel):
    id: str
    type: str
    position: dict


class QuantumVisualSync:
    def __init__(self, vial_id: str):
        self.vial_id = vial_id
    
    def sync_quantum_state(self, vial_id: str) -> dict:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        backend = Aer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=1).result()
        return {"vial_id": vial_id, "quantum_state": str(result.get_counts())}
    
    def create_quantum_circuit_from_visual(self, components: list[Component]) -> dict:
        num_qubits = max(len(components), 2)
        qc = QuantumCircuit(num_qubits, num_qubits)
        visualization = {"nodes": [], "edges": []}
        for comp in components:
            if comp.type == "quantum_gate":
                qc.h(0)
            visualization["nodes"].append({"id": comp.id, "label": comp.type, "position": comp.position})
        return {"circuit": str(qc), "visualization": visualization}
