import torch
import torch.nn as nn
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

class VialAgent(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class MCPAlchemist:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGO_URL", "mongodb://mongo:27017/vial"))
        self.db = self.mongo_client.vial
        self.model = VialAgent()

    def setup_local_db(self):
        # Initialize MongoDB indexes
        self.db.agents.create_index("hash", unique=True)

    def train(self, data):
        # Placeholder: Train PyTorch model
        inputs = torch.tensor([float(x) for x in data.split(",")]).reshape(-1, 10)
        outputs = self.model(inputs)
        self.db.agents.insert_one({"hash": "trained_model", "data": data, "status": "trained"})
        return {"status": "training complete", "output": outputs.tolist()}

mcp_alchemist = MCPAlchemist()
