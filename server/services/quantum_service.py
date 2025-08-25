import torch
import torch.nn as nn
from pathlib import Path
from sqlalchemy.orm import Session
from server.models.base import QuantumCircuit, Wallet
from server.database.session import DatabaseManager
import hashlib

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class QuantumService:
    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QuantumNeuralNetwork().to(self.device)
        self.db_manager = DatabaseManager("sqlite:///./test.db")

    async def sync_circuit(self, circuit_id: int) -> Dict:
        with self.db_manager.get_session() as session:
            circuit = session.query(QuantumCircuit).filter_by(id=circuit_id).first()
            if not circuit:
                raise ValueError("Circuit not found")
            input_data = torch.tensor([float(x) for x in circuit.qasm_code.split()[:8]], device=self.device)
            output = self.model(input_data)
            return {"circuit_id": circuit_id, "output": output.tolist()}

    async def get_wallet_balance(self, wallet_id: str) -> float:
        with self.db_manager.get_session() as session:
            wallet = session.query(Wallet).filter_by(id=wallet_id).first()
            if not wallet:
                raise ValueError("Wallet not found")
            return wallet.balance

    async def save_model(self, model_name: str):
        torch.save(self.model.state_dict(), self.model_dir / f"{model_name}.pth")
