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
        # New .md format (compatible with vial_wallet_export_2025-08-21T21-33-10-827Z.md)
        new_format = {
            "Wallet": {
                "Wallet Key": user_id,
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
        # Old markdown format (compatible with webxos_wallet.py)
        old_format = (
            f"# WEBXOS Wallet\n\nNetwork ID: {user_id}\nBalance: {wallet.balance} $WEBXOS\n\n## Transactions\n"
        )
        for tx in wallet.transactions:
            old_format += f"- {tx['timestamp']}: {tx.get('amount', 0.0)} $WEBXOS (ID: {tx.get('id', str(uuid.uuid4()))})\n"
        return {"status": "exported", "new_format": new_format, "old_format": old_format}
    except Exception as e:
        os.makedirs("db", exist_ok=True)
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.utcnow().isoformat()}]** Export error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/import")
async def import_wallet(wallet_data: dict, token: str = Depends(verify_jwt)):
    try:
        # Handle new .md format
        user_id = wallet_data.get("Wallet", {}).get("Wallet Key")
        if user_id:
            transaction = {
                "action": "import",
                "timestamp": datetime.utcnow().isoformat(),
                "data": wallet_data
            }
            result = await wallet_manager.update_wallet(user_id, transaction)
            return {"status": "imported", "wallet": result.dict()}
        # Handle old markdown format
        network_id = wallet_data.get("Network ID")
        if network_id:
            balance = float(wallet_data.get("Balance", "0.0").replace(" $WEBXOS", ""))
            transactions = [
                {
                    "id": tx.get("id", str(uuid.uuid4())),
                    "amount": float(tx["amount"]),
                    "timestamp": tx["timestamp"]
                }
                for tx in wallet_data.get("Transactions", [])
            ]
            transaction = {
                "action": "import",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"balance": balance, "transactions": transactions}
            }
            result = await wallet_manager.update_wallet(network_id, transaction)
            return {"status": "imported", "wallet": result.dict()}
        raise HTTPException(status_code=400, detail="Invalid wallet data: missing Wallet Key or Network ID")
    except Exception as e:
        os.makedirs("db", exist_ok=True)
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.utcnow().isoformat()}]** Import error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
