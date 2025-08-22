from fastapi import APIRouter, WebSocket
from server.logging import logger
import json

router = APIRouter()
connected_clients = []


async def broadcast_message(message: str):
    for client in connected_clients:
        try:
            await client.send_json({"message": message})
        except Exception as e:
            logger.log(f"Broadcast error: {str(e)}")
            connected_clients.remove(client)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            vial_id = data.get("vial_id")
            message = data.get("message")
            if vial_id and message:
                await broadcast_message(f"{vial_id}: {message}")
                logger.log(f"WebSocket message from {vial_id}: {message}")
                await websocket.send_json({
                    "status": "sent",
                    "vial_id": vial_id
                })
            else:
                await websocket.send_json({"error": "Invalid data"})
    except Exception as e:
        logger.log(f"WebSocket error: {str(e)}")
        connected_clients.remove(websocket)
        await websocket.close()
