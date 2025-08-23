from redis import Redis
from pymongo import MongoClient
from server.logging import logger
from server.services.mcp_alchemist import Alchemist
import uuid
import os

class NetworkSync:
    def __init__(self):
        self.redis = Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]
        self.alchemist = Alchemist()

    async def sync_vial_states(self, network_id: str, vial_states: dict) -> dict:
        request_id = str(uuid.uuid4())
        try:
            self.redis.set(f"vial:{network_id}", str(vial_states))
            self.db.vial_states.update_one(
                {"network_id": network_id},
                {"$set": {"states": vial_states, "timestamp": "2025-08-23T01:00:00Z"}},
                upsert=True
            )
            results = []
            for vial_id, state in vial_states.items():
                result = await self.alchemist.delegate_task(
                    "vial_status_get",
                    {"params": {"vial_id": vial_id}}
                )
                results.append({"vial_id": vial_id, "result": result})
                logger.info(f"Synced vial {vial_id} for {network_id}", request_id=request_id)
            self.db.sync_logs.insert_one({
                "network_id": network_id,
                "results": results,
                "timestamp": "2025-08-23T01:00:00Z"
            })
            return {"status": "synced", "results": results, "request_id": request_id}
        except Exception as e:
            logger.error(f"Sync error: {str(e)}", request_id=request_id)
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-23T01:00:00Z]** Sync error: {str(e)}\n")
            raise
