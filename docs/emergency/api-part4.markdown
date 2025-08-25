# WebXOS 2025 Vial MCP SDK: API Emergency Backup - Part 4 (Wallet Rebuild and Security)

**Objective**: Rebuild wallet functionality with enhanced security agents for rate limiting, audit logging, and intrusion detection.

**Instructions for LLM**:
1. Update `server/webxos_wallet.py` with enhanced security features.
2. Create `server/security/security_agent.py` for rate limiting and logging.
3. Integrate with `server/main.py` and database models.
4. Ensure AES-256-GCM encryption and input sanitization.

## Step 1: Update Wallet and Security Files

### server/webxos_wallet.py
```python
from server.security.crypto_engine import CryptoEngine
from server.models.base import Wallet, Session
import re
import json
from typing import Dict
from pydantic import BaseModel, validator

class WalletModel(BaseModel):
    address: str
    private_key: str
    balance: float

    @validator('address')
    def validate_address(cls, v):
        if not re.match(r'^0x[a-fA-F0-9]{40}$', v):
            raise ValueError('Invalid wallet address')
        return v

    @validator('private_key')
    def validate_private_key(cls, v):
        if not re.match(r'^[a-fA-F0-9]{64}$', v):
            raise ValueError('Invalid private key')
        return v

class WebXOSWallet:
    def __init__(self, password: str):
        self.crypto = CryptoEngine(password)
        self.session = Session()

    def create_wallet(self, address: str, private_key: str, balance: float = 0.0) -> Wallet:
        wallet = Wallet(address=address, private_key=private_key, balance=balance)
        self.session.add(wallet)
        self.session.commit()
        return wallet

    def get_wallet(self, address: str) -> Wallet:
        address = self.sanitize_input(address)
        wallet = self.session.query(Wallet).filter_by(address=address).first()
        if not wallet:
            raise ValueError("Wallet not found")
        return wallet

    def export_wallet(self, address: str, filename: str):
        wallet = self.get_wallet(address)
        wallet_data = {"address": wallet.address, "private_key": wallet.decrypt_private_key(), "balance": wallet.balance}
        with open(filename, 'w') as f:
            f.write(json.dumps(wallet_data, indent=2))
        self.crypto.save_key(f"{filename}.key")

    def import_wallet(self, filename: str):
        with open(filename, 'r') as f:
            wallet_data = json.load(f)
        wallet = Wallet(**wallet_data)
        self.session.add(wallet)
        self.session.commit()
        return wallet

    def sanitize_input(self, input_str: str) -> str:
        return re.sub(r'[<>{};]', '', input_str)
```

### server/security/security_agent.py
```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2AuthorizationCodeBearer
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logging.basicConfig(filename='audit.log', level=logging.INFO)

class SecurityAgent:
    def __init__(self):
        self.rate_limit = defaultdict(list)
        self.max_requests = 100
        self.window_seconds = 60

    def rate_limit_check(self, user_id: str):
        now = datetime.utcnow()
        self.rate_limit[user_id] = [t for t in self.rate_limit[user_id] if now - t < timedelta(seconds=self.window_seconds)]
        if len(self.rate_limit[user_id]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        self.rate_limit[user_id].append(now)

    def log_access(self, endpoint: str, user_id: str, status: int):
        logging.info(f"[{datetime.utcnow()}] Endpoint: {endpoint}, User: {user_id}, Status: {status}")

    def detect_intrusion(self, request_data: str):
        suspicious_patterns = [r'\b(union|select|drop|alter)\b', r'[<>{};]', r'exec\s*\(']
        for pattern in suspicious_patterns:
            if re.search(pattern, request_data, re.IGNORECASE):
                logging.warning(f"[{datetime.utcnow()}] Suspicious request detected: {request_data}")
                raise HTTPException(status_code=403, detail="Suspicious request detected")

security_agent = SecurityAgent()

async def secure_endpoint(token: dict = Depends(OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
    scopes={"openid": "OpenID", "email": "Email", "profile": "Profile"}
))):
    security_agent.rate_limit_check(token.get("sub", "unknown"))
    security_agent.detect_intrusion(str(token))
    return token
```

## Step 2: Update Main Application
Update `server/main.py`:
```python
from server.security.security_agent import secure_endpoint
# Replace verify_token with secure_endpoint in all routes
```

## Step 3: Validation
```bash
curl -H "Authorization: Bearer <token>" -X POST http://localhost:8000/mcp/wallet/create -d '{"address": "0x1234567890abcdef1234567890abcdef12345678", "private_key": "a"*64}'
cat audit.log
```

**Next**: Proceed to `api-part5.md` for SDK rebuild guide.