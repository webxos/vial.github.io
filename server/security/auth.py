# server/security/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import os

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_wallet(token: str = Depends(oauth2_scheme)):
    """Validate OAuth2 token and return wallet."""
    try:
        if token != os.getenv("VERCEL_TOKEN"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        with SessionLocal() as session:
            wallet = session.query(Wallet).filter_by(
                address="e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d"
            ).first()
            if not wallet:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Wallet not found"
                )
        return wallet
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}"
        )
