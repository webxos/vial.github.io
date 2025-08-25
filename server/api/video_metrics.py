from fastapi import APIRouter, Depends, Response
from prometheus_client import generate_latest, Counter, Histogram
from server.security.oauth2 import validate_token
from server.wasm.video_processor import WASMVideoProcessor

router = APIRouter(prefix="/metrics/video")
video_processor = WASMVideoProcessor()

VIDEO_FRAMES_PROCESSED = Counter(
    'mcp_video_frames_processed_total',
    'Total video frames processed',
    ['resolution']
)

VIDEO_PROCESSING_LATENCY = Histogram(
    'mcp_video_processing_duration_seconds',
    'Video processing latency',
    ['resolution']
)

@router.get("")
async def video_metrics(token: str = Depends(validate_token)):
    """Expose video processing metrics."""
    # Simulate metric updates (replace with actual data in production)
    VIDEO_FRAMES_PROCESSED.labels(resolution="1080p").inc()
    VIDEO_PROCESSING_LATENCY.labels(resolution="1080p").observe(0.3)
    
    return Response(generate_latest(), media_type="text/plain")
