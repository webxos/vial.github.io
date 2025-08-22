from fastapi import FastAPI, Request
from server.services.advanced_logging import AdvancedLogger
import time


logger = AdvancedLogger()
analytics_data = {}


def setup_usage_analytics(app: FastAPI):
    @app.middleware("http")
    async def track_usage(request: Request, call_next):
        endpoint = request.url.path
        analytics_data.setdefault(endpoint, {"count": 0, "last_access": None})
        analytics_data[endpoint]["count"] += 1
        analytics_data[endpoint]["last_access"] = time.time()
        logger.log("Usage tracked", extra={"endpoint": endpoint, "count": analytics_data[endpoint]["count"]})
        return await call_next(request)
    
    @app.get("/analytics")
    async def get_analytics():
        logger.log("Analytics retrieved", extra={"endpoints": list(analytics_data.keys())})
        return {"analytics": analytics_data}
