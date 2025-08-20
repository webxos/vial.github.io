from server.services.mongodb_handler import mongodb_handler
import uuid
from server.config import get_settings

class WebXOSWallet:
    def __init__(self):
        self.settings = get_settings()
        self.handler = mongodb_handler

    def create_wallet(self, user_id: str):
        wallet = {
            "address": str(uuid.uuid4()),
            "user_id": user_id,
            "balance": 0.0,
            "hash": str(uuid.uuid4())
        }
        wallet_id = self.handler.save_wallet(wallet)
        return wallet

    def get_balance(self, address: str):
        wallet = self.handler.get_wallet(address)
        if not wallet:
            raise ValueError("Wallet not found")
        return wallet["balance"]

    def update_balance(self, address: str, amount: float):
        wallet = self.handler.get_wallet(address)
        if not wallet:
            raise ValueError("Wallet not found")
        wallet["balance"] += amount
        self.handler.db.wallets.update_one({"address": address}, {"$set": {"balance": wallet["balance"]}})
        return wallet["balance"]

    def export_wallet(self, address: str):
        wallet = self.handler.get_wallet(address)
        if not wallet:
            raise ValueError("Wallet not found")
        return {
            "address": wallet["address"],
            "balance": wallet["balance"],
            "hash": wallet["hash"]
        }

    def import_wallet(self, wallet_data: dict):
        if not all(key in wallet_data for key in ["address", "balance", "hash"]):
            raise ValueError("Invalid wallet data")
        self.handler.save_wallet(wallet_data)
        return wallet_data["address"]

webxos_wallet = WebXOSWallet()
