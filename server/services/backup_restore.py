import json
import os
import time
from server.services.mongodb_handler import MongoDBHandler
from server.services.database import get_db
from server.models.webxos_wallet import Wallet
from sqlalchemy import select
from fastapi import HTTPException


class BackupRestore:
    def __init__(self):
        self.mongo = MongoDBHandler()
        self.backup_dir = "/app/backups"

    async def backup_data(self):
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            backup_file = f"{self.backup_dir}/backup_{int(time.time())}.json"
            data = []
            async for doc in self.mongo.collection.find():
                doc["_id"] = str(doc["_id"])
                data.append(doc)
            with open(backup_file, "w") as f:
                json.dump(data, f)
            async with get_db() as db:
                wallets = await db.execute(select(Wallet))
                with open(f"{self.backup_dir}/wallets_{int(time.time())}.json", "w") as f:
                    json.dump([w.__dict__ for w in wallets.scalars()], f)
            return {"status": "backup_completed", "file": backup_file}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

    async def restore_data(self, backup_file: str):
        try:
            if not os.path.exists(backup_file):
                raise HTTPException(status_code=404, detail="Backup file not found")
            with open(backup_file, "r") as f:
                data = json.load(f)
            await self.mongo.collection.delete_many({})
            for doc in data:
                await self.mongo.save_metadata(doc)
            return {"status": "restore_completed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")
