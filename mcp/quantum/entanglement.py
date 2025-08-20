from .quantum_state import quantum_state
from datetime import datetime

class EntanglementManager:
    def __init__(self):
        self.links = {}

    def entangle_qubits(self, link_id: str, qubit_count: int):
        state = quantum_state.initialize_qubits(qubit_count)
        self.links[link_id] = {
            "state": state,
            "last_update": datetime.now().isoformat()
        }
        return {"link_id": link_id, "status": "entangled", "time": "06:07 PM EDT, Aug 20, 2025"}

entanglement_manager = EntanglementManager()
