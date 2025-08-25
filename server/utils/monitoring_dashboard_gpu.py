from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from fastapi.responses import Response
import torch
import pynvml

pynvml.nvmlInit()
gpu_utilization = Gauge('webxos_gpu_utilization', 'GPU Utilization Percentage', ['gpu_id'])
gpu_memory = Gauge('webxos_gpu_memory', 'GPU Memory Usage (MB)', ['gpu_id'])

class MonitoringDashboardGPU:
    def __init__(self):
        self.request_counter = Counter('webxos_requests_total', 'Total API requests', ['endpoint'])
        self.request_duration = Histogram('webxos_request_duration_seconds', 'Request duration in seconds', ['endpoint'])
        self.update_gpu_metrics()

    def update_gpu_metrics(self):
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
            gpu_utilization.labels(gpu_id=i).set(util)
            gpu_memory.labels(gpu_id=i).set(mem)

    def record_metrics(self, endpoint: str, duration: float):
        self.request_counter.labels(endpoint=endpoint).inc()
        self.request_duration.labels(endpoint=endpoint).observe(duration)
        self.update_gpu_metrics()

monitoring_dashboard_gpu = MonitoringDashboardGPU()

router = APIRouter(prefix="/mcp/monitoring", tags=["monitoring"])

@router.get("/metrics")
async def get_metrics(token: dict = Depends(verify_token)):
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
