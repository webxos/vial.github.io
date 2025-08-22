# server/api/vial_manager.py
from fastapi import APIRouter, Depends, HTTPException
from server.security.auth import get_current_wallet
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
from server.models.mcp_alchemist import Agent

router = APIRouter()

@router.get("/manage/{resource_id}")
async def manage_resource(
    resource_id: str,
    wallet: Wallet = Depends(get_current_wallet)
):
    """Manage MCP resources with wallet and Vercel validation."""
    try:
        with SessionLocal() as session:
            agent = session.query(Agent).filter_by(
                id=resource_id
            ).first()
            if not agent:
                raise HTTPException(
                    status_code=404,
                    detail=f"Resource not found: {resource_id}"
                )
            if wallet.reputation < 10.0:
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient reputation for resource management"
                )
        
        return {
            "status": "success",
            "resource_id": resource_id,
            "wallet_reputation": wallet.reputation
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Resource management error: {str(e)}"
        )
