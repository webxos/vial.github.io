from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.models.dao_repository import DAORepository
from server.models.user_repository import UserRepository
from server.models.transaction_repository import TransactionRepository
from server.config.database import get_db
from sqlalchemy.orm import Session
from fastapi.security import OAuth2AuthorizationCodeBearer
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO, filename="logs/dao_endpoint.log")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/mcp/auth/authorize",
    tokenUrl="/mcp/auth/token",
    scheme_name="OAuth2PKCE"
)

class DAORewardRequest(BaseModel):
    wallet_id: str
    points: int
    contribution_type: str  # e.g., "code", "circuit", "docs"

@router.post("/reward")
async def add_reward(request: DAORewardRequest, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        # Validate input
        if request.points < 0:
            raise HTTPException(status_code=400, detail="Points must be non-negative")
        if request.contribution_type not in ["code", "circuit", "docs", "other"]:
            raise HTTPException(status_code=400, detail="Invalid contribution type")
        
        # Verify wallet exists
        user_repo = UserRepository(db)
        user = user_repo.db.query(user_repo.User).filter(user_repo.User.wallet_id == request.wallet_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Wallet not found")
        
        # Add reputation and log transaction
        dao_repo = DAORepository(db)
        dao_repo.add_reputation(request.wallet_id, request.points)
        transaction_repo = TransactionRepository(db)
        transaction_repo.log_transaction(
            wallet_id=request.wallet_id,
            transaction_type="reward",
            amount=request.points,
            description=f"Reward for {request.contribution_type}"
        )
        
        return {
            "wallet_id": request.wallet_id,
            "points_added": request.points,
            "total_points": dao_repo.get_reputation(request.wallet_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"DAO reward error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reputation/{wallet_id}")
async def get_reputation(wallet_id: str, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        dao_repo = DAORepository(db)
        points = dao_repo.get_reputation(wallet_id)
        return {"wallet_id": wallet_id, "reputation_points": points}
    except Exception as e:
        logging.error(f"Reputation fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
