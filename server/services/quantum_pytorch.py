from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
from qiskit import QuantumCircuit, Aer, execute
import torch
import torch.nn as nn
from typing import Dict

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, qubits: int):
        super(QuantumNeuralNetwork, self).__init__()
        self.qubits = qubits
        self.fc1 = nn.Linear(2**qubits, 16)
        self.fc2 = nn.Linear(16, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QuantumPytorchService:
    def __init__(self):
        self.backend = Aer.get_backend("qasm_simulator")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QuantumNeuralNetwork(qubits=2).to(self.device)

    def run_hybrid_circuit(self, qubits: int, layers: int, shots: int = 1000) -> Dict:
        if qubits < 1 or qubits > 4 or layers < 1 or layers > 3:
            raise ValueError("Invalid qubits or layers")
        
        circuit = QuantumCircuit(qubits, qubits)
        for _ in range(layers):
            circuit.h(range(qubits))
            circuit.cz(0, 1)
            circuit.measure_all()
        result = execute(circuit, self.backend, shots=shots).result()
        counts = result.get_counts()
        
        probs = torch.tensor([counts.get(str(i), 0) / shots for i in range(2**qubits)], device=self.device)
        with torch.no_grad():
            prediction = self.model(probs.float())
        return {"counts": counts, "prediction": prediction.tolist()}

quantum_pytorch_service = QuantumPytorchService()

router = APIRouter(prefix="/mcp/quantum-pytorch", tags=["quantum-pytorch"])

@router.get("/hybrid")
async def run_hybrid(qubits: int = 2, layers: int = 1, shots: int = 1000, token: dict = Depends(verify_token)) -> Dict:
    try:
        return quantum_pytorch_service.run_hybrid_circuit(qubits, layers, shots)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
