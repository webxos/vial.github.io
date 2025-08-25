from fastapi import APIRouter, Depends, Response
from prometheus_client import generate_latest, Counter, Histogram
from server.security.oauth2 import validate_token
from obswebsocket import WebSocketClient
import asyncio
import os

router = APIRouter(prefix="/metrics/video_obs")
obs_client = WebSocketClient(os.getenv("OBS_WS_URL"), password=os.getenv("OBS_WS_PASSWORD"))

OBS_FRAMES_STREAMED = Counter(
    'mcp_obs_frames_streamed_total',
    'Total OBS frames streamed',
    ['resolution']
)

OBS_STREAM_LATENCY = Histogram(
    'mcp_obs_stream_duration_seconds',
    'OBS stream latency',
    ['resolution']
)

@router.get("")
async def video_obs_metrics(token: str = Depends(validate_token)):
    """Expose OBS video streaming metrics."""
    await obs_client.connect()
    try:
        # Simulate metric updates (replace with actual OBS data in production)
        OBS_FRAMES_STREAMED.labels(resolution="1080p").inc()
        OBS_STREAM_LATENCY.labels(resolution="1080p").observe(0.4)
        
        return Response(generate_latest(), media_type="text/plain")
    finally:
        await obs_client.disconnect()
