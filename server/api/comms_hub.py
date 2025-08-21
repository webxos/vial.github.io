from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.models.webxos_wallet import WebXOSWallet
from server.services.logging import Logger
from server.services.security import verify_jwt


router = APIRouter()
logger = Logger("comms_hub")
wallet_manager = WebXOSWallet()


class CommsRequest(BaseModel):
    message: str
    network_id: str


@router.post("/comms_hub")
async def comms_hub(request: CommsRequest, token: str = Depends(verify_jwt)):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="No prompt entered")
        # Placeholder for LangChain integration
        response = f"Simulated NanoGPT response to: {request.message}"
        await wallet_manager.update_wallet(
            request.network_id,
            {"action": "comms", "message": request.message}
        )
        await logger.info(f"Comms processed for network {request.network_id}: {request.message}")
        return {"response": response}
    except Exception as e:
        await logger.error(f"Comms error: {str(e)}")
        os.makedirs("db", exist_ok=True)
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.utcnow().isoformat()}]** Comms error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
