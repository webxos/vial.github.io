from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from server.services.advanced_logging import AdvancedLogger


request_counter = Counter("vial_requests_total", "Total API requests", ["endpoint"])
request_duration = Histogram("vial_request_duration_seconds", "Request duration", ["endpoint"])
logger = AdvancedLogger()


def setup_prometheus_metrics(app: FastAPI):
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        endpoint = request.url.path
        request_counter.labels(endpoint=endpoint).inc()
        with request_duration.labels(endpoint=endpoint).time():
            response = await call_next(request)
        logger.log("Metrics recorded", extra={"endpoint": endpoint})
        return response
    
    app.mount("/metrics", make_asgi_app())
    logger.log("Prometheus metrics initialized", extra={"system": "monitoring"})
