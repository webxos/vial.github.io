from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
from server.models.wallet_models import Wallet, Session
from typing import Dict
import numpy as np
from datetime import datetime, timedelta

class EconomicService:
    def __init__(self):
        self.session = Session()

    def forecast_balance(self, address: str, days: int) -> Dict:
        wallet = self.session.query(Wallet).filter_by(address=address).first()
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found")
        
        # Simple linear forecast based on historical balance (mock data)
        historical_balances = [wallet.balance * (1 + i * 0.01) for i in range(-10, 0)]
        forecast = []
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            predicted_balance = np.poly1d(np.polyfit(range(len(historical_balances)), historical_balances, 1))(len(historical_balances) + i)
            forecast.append({"date": date, "balance": max(0, predicted_balance)})
        return {"address": address, "forecast": forecast}

economic_service = EconomicService()

router = APIRouter(prefix="/mcp/economic", tags=["economic"])

@router.get("/forecast/{address}")
async def forecast_balance(address: str, days: int = 7, token: dict = Depends(verify_token)) -> Dict:
    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 30")
    return economic_service.forecast_balance(address, days)
