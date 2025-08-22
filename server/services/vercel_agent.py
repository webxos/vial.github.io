# server/services/vercel_agent.py
import httpx
import logging
from fastapi import Depends, HTTPException
from server.security.auth import oauth2_scheme
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

logger = logging.getLogger(__name__)

async def check_vercel_deployment(token: str = Depends(oauth2_scheme)) -> dict:
    """Check Vercel deployment status and validate wallet."""
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.get(
                "https://api.vercel.com/v6/deployments",
                headers=headers
            )
            response.raise_for_status()
            deployments = response.json()
            with SessionLocal() as session:
                wallet = session.query(Wallet).filter_by(
                    address="e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d"
                ).first()
                if not wallet:
                    raise HTTPException(
                        status_code=404, detail="Wallet not found"
                    )
            logger.info(f"Vercel deployment check: {deployments}")
            return {
                "status": "success",
                "deployments": deployments,
                "wallet_reputation": wallet.reputation
            }
    except httpx.HTTPStatusError as e:
        logger.error(f"Vercel API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
