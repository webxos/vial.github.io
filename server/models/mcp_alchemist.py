from torch import nn, optim
import torch
from qiskit import QuantumCircuit
from server.services.mongodb_handler import MongoDBHandler


class MCPAlchemist(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)
        self.relu = nn.ReLU()
        self.mongo = MongoDBHandler()

    async def predict_quantum_outcome(self, circuit_data: dict):
        circuit = QuantumCircuit(circuit_data.get("qubits", 2))
        for gate in circuit_data.get("gates", []):
            if gate["type"] == "h":
                circuit.h(gate["target"])
            elif gate["type"] == "cx":
                circuit.cx(gate["control"], gate["target"])
        input_tensor = torch.tensor(
            [float(circuit.num_qubits), float(circuit.depth())],
            dtype=torch.float32
        )
        input_tensor = input_tensor.view(1, -1)
        with torch.no_grad():
            output = self.fc2(self.relu(self.fc1(input_tensor)))
        prediction = output.tolist()[0]
        await self.mongo.save_quantum_result(circuit_id=str(id(circuit_data)), result=prediction)
        return {"prediction": prediction}
