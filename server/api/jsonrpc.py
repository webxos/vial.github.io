# server/api/jsonrpc.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from server.services.vial_manager import VialManager
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/jsonrpc")
async def jsonrpc_endpoint(request: Request):
    """Handle JSON-RPC 2.0 requests for wallet and vial operations."""
    try:
        data = await request.json()
        method = data.get("method")
        request_id = data.get("id", 1)

        with SessionLocal() as session:
            if method == "wallet.get_balance":
                wallet_address = data.get("params", {}).get("address")
                wallet = session.query(Wallet).filter_by(address=wallet_address).first()
                if not wallet:
                    raise HTTPException(status_code=404, detail="Wallet not found")
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "result": {"balance": wallet.balance, "address": wallet.address},
                    "id": request_id
                })

            elif method == "vial.train":
                vial_id = data.get("params", {}).get("vial_id")
                vial_manager = VialManager()
                result = await vial_manager.train_vial(vial_id)
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                })

            else:
                raise HTTPException(status_code=400, detail="Method not supported")

    except Exception as e:
        logger.error(f"JSON-RPC error: {str(e)}")
        return JSONResponse({
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": str(e)},
            "id": request_id
        })
