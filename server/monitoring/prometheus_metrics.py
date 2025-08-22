from fastapi import FastAPI
from prometheus_client import Counter, make_asgi_app
from server.services.advanced_logging import AdvancedLogger


logger = AdvancedLogger()
request_counter = Counter("http_requests_total", "Total HTTP Requests", ["endpoint"])


def setup_prometheus_metrics(app: FastAPI):
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    @app.middleware("http")
    async def count_requests(request, call_next):
        request_counter.labels(endpoint=request.url.path).inc()
        logger.log("Request counted",
                   extra={"endpoint": request.url.path})
        return await call_next(request)
    
    logger.log("Prometheus metrics initialized",
               extra={"system": "prometheus"})
