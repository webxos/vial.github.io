# server/security/rbac.py
from fastapi import Depends, HTTPException, status
from server.security.auth import get_current_wallet
from server.models.webxos_wallet import Wallet

async def check_role(
    role: str,
    wallet: Wallet = Depends(get_current_wallet)
):
    """Check if wallet has required role based on reputation."""
    try:
        if wallet.reputation < 20.0:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient reputation for role: {role}"
            )
        return {"status": "success", "role": role}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role check error: {str(e)}"
        )
