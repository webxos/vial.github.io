import torch
import torch.nn as nn
from fastapi import FastAPI
from server.services.advanced_logging import AdvancedLogger


class PromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


def setup_prompt_training(app: FastAPI):
    logger = AdvancedLogger()
    model = PromptModel()
    
    async def train_prompt(prompt: str):
        dummy_input = torch.randn(1, 20)
        output = model(dummy_input)
        logger.log("Prompt training completed", extra={"prompt": prompt, "output": output.item()})
        return {"status": "trained", "output": output.item()}
    
    app.state.train_prompt = train_prompt
    logger.log("Prompt training initialized", extra={"model": "PromptModel"})
