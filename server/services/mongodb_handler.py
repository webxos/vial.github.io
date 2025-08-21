from pymongo import MongoClient
from server.config import settings
from datetime import datetime


class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient(settings.MONGO_URL)
        self.db = self.client["vial_mcp"]
        self.collection = self.db["metadata"]

    async def save_metadata(self, data: dict):
        result = self.collection.insert_one(data)
        return {"status": "saved", "id": str(result.inserted_id)}

    async def get_metadata(self, query: dict):
        result = self.collection.find_one(query)
        if result:
            result["_id"] = str(result["_id"])
            return result
        return None

    async def save_quantum_result(self, circuit_id: str, result: dict):
        result = self.collection.insert_one({
            "circuit_id": circuit_id,
            "result": result,
            "timestamp": datetime.utcnow()
        })
        return {"status": "saved", "id": str(result.inserted_id)}
