from pymongo import MongoClient
from server.config import get_settings
from server.logging import logger


class MongoDBHandler:
    def __init__(self):
        settings = get_settings()
        self.client = MongoClient(settings.MONGO_URL)
        self.db = self.client["vial"]

    def insert(self, collection: str, data: dict):
        try:
            result = self.db[collection].insert_one(data)
            logger.info(f"Inserted data into {collection}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Insert failed: {str(e)}")
            raise ValueError(f"Insert failed: {str(e)}")


mongodb_handler = MongoDBHandler()
