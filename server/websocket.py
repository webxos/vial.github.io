import logging
from fastapi import WebSocket, WebSocketDisconnect
from redis.asyncio import Redis
from server.auth.rbac import check_rbac

logger = logging.getLogger(__name__)

async def websocket_endpoint(websocket: WebSocket, token: str):
    """Handle WebSocket for real-time updates."""
    await websocket.accept()
    try:
        await check_rbac(token, ["websocket:connect"])
        redis = Redis.from_url("redis://localhost:6379/0")
        pubsub = redis.pubsub()
        await pubsub.subscribe("agent_updates", "quantum_status")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_json({"channel": message["channel"], "data": message["data"]})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close(code=1000)
