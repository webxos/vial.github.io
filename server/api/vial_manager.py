from fastapi import APIRouter, Depends, HTTPException
from server.models.webxos_wallet import WebXOSWallet, WalletModel
from server.models.mcp_alchemist import MCPAlchemist
from server.services.logging import Logger
from server.services.security import verify_jwt
import torch
import torch.nn as nn
from typing import Dict


router = APIRouter()
logger = Logger("vial_manager")
wallet_manager = WebXOSWallet()


class VialAgent(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.state = {"trained": False, "parameters": {}}

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

    def train(self, content: str, filename: str):
        self.state["trained"] = True
        self.state["parameters"] = {"content": content, "filename": filename}

    def reset(self):
        self.state = {"trained": False, "parameters": {}}

    def get_state(self):
        return self.state


@router.post("/vials/train")
async def train_vials(network_id: str, content: str, filename: str, token: str = Depends(verify_jwt)):
    try:
        vials = {
            "vial1": VialAgent(10, 1),
            "vial2": VialAgent(10, 2),
            "vial3": VialAgent(15, 3),
            "vial4": VialAgent(25, 4)
        }
        alchemist = MCPAlchemist()
        balance_earned = 0.0
        for vial_name, agent in vials.items():
            agent.train(content, filename)
            transaction = {
                "action": "train_vial",
                "vial": vial_name,
                "filename": filename
            }
            wallet = await wallet_manager.update_wallet(network_id, transaction, balance_increment=0.1)
            balance_earned += wallet.balance
            await alchemist.predict_quantum_outcome({"qubits": [], "entanglement": "synced"})
            await logger.info(f"Trained {vial_name} for network {network_id}")
        return {"status": "trained", "balance_earned": balance_earned, "vials": {name: agent.get_state() for name, agent in vials.items()}}
    except Exception as e:
        await logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vials/reset")
async def reset_vials(network_id: str, token: str = Depends(verify_jwt)):
    try:
        vials = {
            "vial1": VialAgent(10, 1),
            "vial2": VialAgent(10, 2),
            "vial3": VialAgent(15, 3),
            "vial4": VialAgent(25, 4)
        }
        for vial_name, agent in vials.items():
            agent.reset()
            await logger.info(f"Reset {vial_name} for network {network_id}")
        return {"status": "reset"}
    except Exception as e:
        await logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
