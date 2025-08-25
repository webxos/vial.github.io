from fastapi import Depends, HTTPException
from fastapi.security import OAuth2AuthorizationCodeBearer
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import re
from typing import Dict
import numpy as np
from server.models.wallet_models import Session, Wallet

logging.basicConfig(filename='security.log', level=logging.INFO)

class AdvancedSecurity:
    def __init__(self):
        self.rate_limit = defaultdict(list)
        self.max_requests = 100
        self.window_seconds = 60
        self.session = Session()
        self.anomaly_scores = defaultdict(list)

    def rate_limit_check(self, user_id: str):
        now = datetime.utcnow()
        self.rate_limit[user_id] = [t for t in self.rate_limit[user_id] if now - t < timedelta(seconds=self.window_seconds)]
        if len(self.rate_limit[user_id]) >= self.max_requests:
            self.log_anomaly(user_id, "Rate limit exceeded")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        self.rate_limit[user_id].append(now)

    def detect_intrusion(self, request_data: str, user_id: str):
        suspicious_patterns = [r'\b(union|select|drop|alter)\b', r'[<>{};]', r'exec\s*\(']
        for pattern in suspicious_patterns:
            if re.search(pattern, request_data, re.IGNORECASE):
                self.log_anomaly(user_id, f"Suspicious request: {request_data}")
                raise HTTPException(status_code=403, detail="Suspicious request detected")

    def log_anomaly(self, user_id: str, message: str):
        logging.warning(f"[{datetime.utcnow()}] User: {user_id}, {message}")
        self.anomaly_scores[user_id].append(1.0)
        if len(self.anomaly_scores[user_id]) > 10:
            self.anomaly_scores[user_id].pop(0)
        if np.mean(self.anomaly_scores[user_id]) > 0.5:
            self.lock_account(user_id)

    def lock_account(self, user_id: str):
        # Placeholder for account locking logic
        logging.error(f"[{datetime.utcnow()}] Account locked: {user_id}")

    def monitor_wallet_activity(self, address: str) -> Dict:
        wallet = self.session.query(Wallet).filter_by(address=address).first()
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found")
        # Mock anomaly detection on balance changes
        score = np.random.random()
        if score > 0.8:
            self.log_anomaly("system", f"Anomalous wallet activity: {address}, score: {score}")
        return {"address": address, "anomaly_score": score}

advanced_security = AdvancedSecurity()

async def secure_endpoint(token: dict = Depends(OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
    scopes={"openid": "OpenID", "email": "Email", "profile": "Profile"}
))):
    user_id = token.get("sub", "unknown")
    advanced_security.rate_limit_check(user_id)
    advanced_security.detect_intrusion(str(token), user_id)
    return token

router = APIRouter(prefix="/mcp/security", tags=["security"])

@router.get("/monitor/{address}")
async def monitor_wallet(address: str, token: dict = Depends(secure_endpoint)) -> Dict:
    return advanced_security.monitor_wallet_activity(address)
