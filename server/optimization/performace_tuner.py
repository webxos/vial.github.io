from fastapi import FastAPI
from server.services.advanced_logging import AdvancedLogger
import time


logger = AdvancedLogger()


def setup_performance_tuner(app: FastAPI):
    @app.middleware("http")
    async def measure_performance(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        logger.log("Request processed",
                   extra={"endpoint": request.url.path,
                          "duration_ms": duration * 1000})
        return response
