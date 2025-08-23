from fastapi import APIRouter, Depends
from prometheus_client import Counter, Histogram
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1/monitoring", tags=["monitoring"])

request_counter = Counter("mcp_requests_total", "Total MCP requests", ["endpoint"])
request_duration = Histogram("mcp_request_duration_seconds", "Request duration", ["endpoint"])


@router.get("/metrics")
async def get_metrics(memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        request_counter.labels(endpoint="/metrics").inc()
        with request_duration.labels(endpoint="/metrics").time():
            session = await memory_manager.get_session("metrics", request_id)
            metrics = {
                "requests_total": request_counter._metrics,
                "request_duration": request_duration._metrics,
                "session_active": bool(session)
            }
            logger.info(f"Metrics retrieved: {metrics}", request_id=request_id)
            return metrics
    except Exception as e:
        logger.error(f"Metrics retrieval error: {str(e)}", request_id=request_id)
        raise


@router.get("/agentic_search_metrics")
async def agentic_search_metrics(memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        request_counter.labels(endpoint="/agentic_search_metrics").inc()
        with request_duration.labels(endpoint="/agentic_search_metrics").time():
            session = await memory_manager.get_session("agentic_search", request_id)
            metrics = {
                "search_requests": request_counter._metrics.get("/agentic_search", 0),
                "session_active": bool(session),
                "sources_analyzed": session.get("sources_analyzed", 0) if session else 0
            }
            logger.info(f"Agentic search metrics retrieved: {metrics}", request_id=request_id)
            return metrics
    except Exception as e:
        logger.error(f"Agentic search metrics error: {str(e)}", request_id=request_id)
        raise
