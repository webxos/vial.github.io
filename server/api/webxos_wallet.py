from fastapi import APIRouter, Depends, HTTPException
from server.models.webxos_wallet import WebXOSWallet, WalletModel
from server.services.security import verify_jwt
import json
import uuid
from datetime import datetime


router = APIRouter()
wallet_manager = WebXOSWallet()


@router.post("/export")
async def export_wallet(user_id: str, token: str = Depends(verify_jwt)):
    try:
        wallet = await wallet_manager.update_wallet(user_id, {"action": "export"})
        export_data = {
            "Wallet": {
                "Wallet Key": str(uuid.uuid4()),
                "Session Balance": f"{wallet.balance:.4f} $WEBXOS",
                "Address": str(uuid.uuid4()),
                "Hash": str(uuid.uuid4()).replace("-", "")
            },
            "API Credentials": {
                "Key": str(uuid.uuid4()),
                "Secret": str(uuid.uuid4()).replace("-", "")
            },
            "Instructions": {
                "Reuse": "Import this .md file via the 'Import' button to resume training.",
                "Extend": "Modify agent code externally, then reimport.",
                "Share": "Send this .md file to others to continue training with the same wallet.",
                "API": "Use API credentials with LangChain to train vials (online mode only).",
                "Cash Out": "$WEBXOS balance and reputation are tied to the wallet address and hash for secure verification (online mode only)."
            }
        }
        return {"status": "exported", "wallet": export_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/import")
async def import_wallet(wallet_data: dict, token: str = Depends(verify_jwt)):
    try:
        user_id = wallet_data.get("Wallet", {}).get("Wallet Key")
        if not user_id:
            raise HTTPException(status_code=400, detail="Invalid wallet data: missing Wallet Key")
        transaction = {
            "action": "import",
            "timestamp": datetime.utcnow().isoformat(),
            "data": wallet_data
        }
        result = await wallet_manager.update_wallet(user_id, transaction)
        return {"status": "imported", "wallet": result.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
