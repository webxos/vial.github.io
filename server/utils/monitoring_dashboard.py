from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from fastapi.responses import Response
from server.services.telescope_service import TelescopeService
from server.services.rag_advanced import RAGService

request_counter = Counter('webxos_requests_total', 'Total API requests', ['endpoint'])
request_duration = Histogram('webxos_request_duration_seconds', 'Request duration in seconds', ['endpoint'])

class MonitoringDashboard:
    def __init__(self):
        self.telescope_service = TelescopeService()
        self.rag_service = RAGService()

    def record_metrics(self, endpoint: str, duration: float):
        request_counter.labels(endpoint=endpoint).inc()
        request_duration.labels(endpoint=endpoint).observe(duration)

monitoring_dashboard = MonitoringDashboard()

router = APIRouter(prefix="/mcp/monitoring", tags=["monitoring"])

@router.get("/metrics")
async def get_metrics(token: dict = Depends(verify_token)):
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
