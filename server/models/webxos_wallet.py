from hashlib import sha256
from .quantum_sync import quantum_sync
from .config import config

class WebXOSWallet:
    def __init__(self):
        self.balance = 0.0000
        self.reputation = 0
        self.agents = ["agent1", "agent2", "agent3", "agent4"]  # 4x agents from earlier vial.html
        self.wallet_id = sha256("webxos_dao".encode()).hexdigest()

    async def update_wallet(self, node_id: str, amount: float):
        self.balance += amount
        await quantum_sync.sync_node(node_id, {"wallet_id": self.wallet_id, "balance": self.balance})
        return {"status": "updated", "balance": self.balance}

    def get_wallet_status(self):
        return {"wallet_id": self.wallet_id, "balance": self.balance, "reputation": self.reputation, "agents": self.agents}

webxos_wallet = WebXOSWallet()
