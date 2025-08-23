from pymongo import MongoClient
from server.logging import logger
import uuid
import os

class ReputationLogger:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]

    async def log_reward(self, vial_id: str, amount: float, reason: str):
        request_id = str(uuid.uuid4())
        try:
            self.db.rewards.insert_one({
                "vial_id": vial_id,
                "amount": amount,
                "reason": reason,
                "request_id": request_id,
                "timestamp": int(__import__('time').time())
            })
            logger.log(
                f"Reward logged for vial {vial_id}: {amount} WebXOS for {reason}",
                request_id=request_id
            )
            return {"status": "success", "request_id": request_id}
        except Exception as e:
            logger.log(f"Reward logging error: {str(e)}", request_id=request_id)
            raise

    async def get_reward_history(self, vial_id: str):
        request_id = str(uuid.uuid4())
        try:
            history = list(self.db.rewards.find({"vial_id": vial_id}))
            logger.log(f"Reward history retrieved for vial {vial_id}", request_id=request_id)
            return history
        except Exception as e:
            logger.log(f"Reward history error: {str(e)}", request_id=request_id)
            raise
