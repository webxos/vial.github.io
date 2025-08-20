import torch
import torch.nn as nn

class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x):
        return torch.relu(self.layer(x))

model = PyTorchModel()
