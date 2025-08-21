from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from server.models.webxos_wallet import WebXOSWallet
from server.api.vial_manager import VialAgent
from server.services.logging import Logger
from server.services.security import verify_jwt
import asyncio
import json


router = APIRouter()
logger = Logger("stream")
wallet_manager = WebXOSWallet()


@router.get("/stream/{network_id}")
async def stream_vials(network_id: str, token: str = Depends(verify_jwt)):
    try:
        vials = {
            "vial1": VialAgent(10, 1),
            "vial2": VialAgent(10, 2),
            "vial3": VialAgent(15, 3),
            "vial4": VialAgent(25, 4)
        }
        async def event_generator():
            for _ in range(5):
                vial_states = {name: agent.get_state() for name, agent in vials.items()}
                await wallet_manager.update_wallet(
                    network_id,
                    {"action": "stream", "vials": list(vials.keys())}
                )
                yield f"data: {json.dumps(vial_states)}\n\n"
                await asyncio.sleep(1)
        await logger.info(f"Streaming vials for network {network_id}")
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        await logger.error(f"Stream error: {str(e)}")
        os.makedirs("db", exist_ok=True)
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.utcnow().isoformat()}]** Stream error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
