from fastapi import APIRouter, WebSocket
from server.services.notification import send_notification


router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            if action == "subscribe":
                await send_notification(
                    "Subscribed to real-time updates",
                    channel="in-app"
                )
                await websocket.send_json({"status": "subscribed"})
            elif action == "quantum_update":
                await websocket.send_json({
                    "status": "quantum_update",
                    "message": "Quantum circuit result available"
                })
            else:
                await websocket.send_json({"error": "Invalid action"})
    except Exception as e:
        await websocket.close()
        await send_notification(f"WebSocket error: {str(e)}", channel="in-app")
