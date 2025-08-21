from pydantic import BaseModel
from server.services.database import get_db
from server.models.webxos_wallet import Wallet
from sqlalchemy import select
from fastapi import HTTPException


class AuthAgent:
    async def assign_role(self, user_id: str, role: str):
        async with get_db() as db:
            wallet = await db.execute(
                select(Wallet).filter_by(user_id=user_id)
            )
            wallet = wallet.scalar_one_or_none()
            if not wallet:
                raise HTTPException(status_code=404, detail="User not found")
            wallet.role = role
            await db.commit()
            return {"user_id": user_id, "role": role}

    async def check_role(self, user_id: str, required_role: str):
        async with get_db() as db:
            wallet = await db.execute(
                select(Wallet).filter_by(user_id=user_id)
            )
            wallet = wallet.scalar_one_or_none()
            if not wallet or wallet.role != required_role:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return True
