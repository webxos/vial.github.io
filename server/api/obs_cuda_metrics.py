from fastapi import APIRouter, Depends, Response
from prometheus_client import generate_latest, Counter, Histogram
from server.security.oauth2 import validate_token
import os
from obswebsocket import WebSocketClient

router = APIRouter(prefix="/cuda/test/obs")

OBS_CUDA_FRAMES = Counter(
    'mcp_obs_cuda_frames_total',
    'Total OBS CUDA frames processed',
    ['resolution']
)

OBS_CUDA_LATENCY = Histogram(
    'mcp_obs_cuda_latency_seconds',
    'OBS CUDA processing latency',
    ['resolution']
)

@router.get("")
async def obs_cuda_metrics(token: str = Depends(validate_token)):
    obs_client = WebSocketClient(os.getenv("OBS_WS_URL"), password=os.getenv("OBS_WS_PASSWORD"))
    await obs_client.connect()
    OBS_CUDA_FRAMES.labels(resolution="1080p").inc()
    OBS_CUDA_LATENCY.labels(resolution="1080p").observe(0.3)
    await obs_client.disconnect()
    return Response(generate_latest(), media_type="text/plain")
