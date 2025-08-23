from prometheus_client import Counter, Histogram, start_http_server
from server.logging_config import logger
import uuid

REQUEST_COUNT = Counter('vial_mcp_requests_total', 'Total API requests', ['endpoint'])
REQUEST_LATENCY = Histogram('vial_mcp_request_latency_seconds', 'API request latency', ['endpoint'])

class PrometheusMetrics:
    def __init__(self):
        start_http_server(9090)
        logger.info("Prometheus metrics server started on port 9090", request_id=str(uuid.uuid4()))

    def track_request(self, endpoint: str):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                request_id = str(uuid.uuid4())
                REQUEST_COUNT.labels(endpoint=endpoint).inc()
                with REQUEST_LATENCY.labels(endpoint=endpoint).time():
                    result = await func(*args, **kwargs)
                logger.info(f"Tracked request for {endpoint}", request_id=request_id)
                return result
            return wrapper
        return decorator
