```python
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict

class QuantumCircuitOptimizer(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class ModelManager:
    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self.models: Dict[str, nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def load_model(self, model_name: str) -> nn.Module:
        if model_name not in self.models:
            model_path = self.model_dir / f"{model_name}.pth"
            model = QuantumCircuitOptimizer()
            if model_path.exists():
                model.load_state_dict(torch.load(model_path))
            model.to(self.device)
            self.models[model_name] = model
        return self.models[model_name]

    async def inference(self, model_name: str, input_data: torch.Tensor) -> torch.Tensor:
        model = await self.load_model(model_name)
        model.eval()
        with torch.no_grad():
            return model(input_data.to(self.device))
