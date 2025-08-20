from pymongo import MongoClient
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGO_URL", "mongodb://mongo:27017/vial"))
        self.db = self.client.vial

    def save_wallet(self, wallet_data: dict):
        wallet_data["hash"] = wallet_data.get("hash", str(uuid.uuid4()))
        self.db.wallets.insert_one(wallet_data)
        return wallet_data["hash"]

    def get_wallet(self, address: str):
        return self.db.wallets.find_one({"address": address})

    def save_agent(self, agent_data: dict):
        agent_data["hash"] = agent_data.get("hash", str(uuid.uuid4()))
        self.db.agents.insert_one(agent_data)
        return agent_data["hash"]

    def get_agent(self, hash: str):
        return self.db.agents.find_one({"hash": hash})

mongodb_handler = MongoDBHandler()
