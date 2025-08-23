import uuid
from server.logging_config import logger

class UsageAnalytics:
    def __init__(self):
        self.request_counts = {}

    def track_request(self, endpoint: str, user_id: str) -> str:
        request_id = str(uuid.uuid4())
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
        logger.info(f"Tracked request to {endpoint} by user {user_id}", request_id=request_id)
        return request_id

    def get_analytics(self, endpoint: str) -> dict:
        request_id = str(uuid.uuid4())
        count = self.request_counts.get(endpoint, 0)
        logger.info(f"Retrieved analytics for {endpoint}: {count} requests", request_id=request_id)
        return {"endpoint": endpoint, "request_count": count, "request_id": request_id}
