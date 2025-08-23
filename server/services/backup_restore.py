from fastapi import Depends
from pymongo import MongoClient
from server.logging import logger
import uuid
import os

class BackupRestoreService:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]

    async def backup_data(self, collection_name: str):
        request_id = str(uuid.uuid4())
        try:
            collection = self.db[collection_name]
            data = list(collection.find())
            backup_path = os.path.join("backups", f"{collection_name}_{request_id}.json")
            os.makedirs("backups", exist_ok=True)
            with open(backup_path, "w") as f:
                import json
                json.dump(data, f, indent=2)
            logger.log(f"Backup created for {collection_name} at {backup_path}", request_id=request_id)
            return {"status": "success", "path": backup_path, "request_id": request_id}
        except Exception as e:
            logger.log(f"Backup error for {collection_name}: {str(e)}", request_id=request_id)
            raise

    async def restore_data(self, collection_name: str, backup_path: str):
        request_id = str(uuid.uuid4())
        try:
            collection = self.db[collection_name]
            collection.drop()
            with open(backup_path, "r") as f:
                import json
                data = json.load(f)
            collection.insert_many(data)
            logger.log(f"Data restored to {collection_name} from {backup_path}", request_id=request_id)
            return {"status": "success", "request_id": request_id}
        except Exception as e:
            logger.log(f"Restore error for {collection_name}: {str(e)}", request_id=request_id)
            raise
