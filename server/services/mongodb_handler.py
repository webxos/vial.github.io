from pymongo import MongoClient
from ..config import config
from ..logging import logger

class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['vial_mcp']
        self.collection = self.db['alchemist_data']
        logger.info("MongoDB Handler initialized for Alchemist")

    def save_training_data(self, data: dict):
        result = self.collection.insert_one(data)
        return str(result.inserted_id)

    def get_training_data(self, query: dict):
        return list(self.collection.find(query))

    def save_wallet_data(self, wallet_id: str, data: dict):
        self.collection.update_one({"wallet_id": wallet_id}, {"$set": data}, upsert=True)

mongodb_handler = MongoDBHandler()
