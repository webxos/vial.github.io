# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 2 (Security Setup)

**Objective**: Implement OAuth2 with PKCE, AES-256-GCM encryption for wallets, and input sanitization for the WebXOS backend.

**Instructions for LLM**:
1. Create security-related files in `server/security/` and `server/api/`.
2. Implement OAuth2 with PKCE for authentication.
3. Set up AES-256-GCM encryption for wallet private keys.
4. Ensure input sanitization to prevent injection attacks.
5. Integrate with `server/main.py`.

## Step 1: Create Security Files

### server/security/crypto_engine.py
```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64

class CryptoEngine:
    def __init__(self, password: str = None):
        if password:
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self.key = kdf.derive(password.encode())
            self.salt = salt
        else:
            self.key = os.urandom(32)
            self.salt = None
        self.iv = os.urandom(12)

    def encrypt(self, data: str) -> str:
        cipher = Cipher(algorithms.AES(self.key), modes.GCM(self.iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data.encode()) + encryptor.finalize()
        return base64.b64encode(self.iv + ciphertext + encryptor.tag).decode()

    def decrypt(self, encrypted_data: str) -> str:
        data = base64.b64decode(encrypted_data)
        iv, ciphertext, tag = data[:12], data[12:-16], data[-16:]
        cipher = Cipher(algorithms.AES(self.key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        return (decryptor.update(ciphertext) + decryptor.finalize()).decode()

    def save_key(self, filename: str):
        with open(filename, 'wb') as f:
            if self.salt:
                f.write(self.salt + self.key)
            else:
                f.write(self.key)
```

### server/api/auth_endpoint.py
```python
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import jwt, JWTError
import httpx
import os
import base64
import hashlib
from datetime import datetime, timedelta

router = APIRouter(prefix="/mcp/auth", tags=["auth"])
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
    scopes={"openid": "OpenID", "email": "Email", "profile": "Profile"}
)

async def verify_token(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, os.getenv("OAUTH_CLIENT_SECRET"), algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.get("/login")
async def login():
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")
    auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={os.getenv('OAUTH_CLIENT_ID')}&"
        f"redirect_uri={os.getenv('OAUTH_REDIRECT_URI')}&"
        f"response_type=code&"
        f"scope=openid%20email%20profile&"
        f"code_challenge={code_challenge}&"
        f"code_challenge_method=S256"
    )
    return {"auth_url": auth_url, "code_verifier": code_verifier}

@router.post("/token")
async def exchange_code(code: str, code_verifier: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": os.getenv("OAUTH_CLIENT_ID"),
                "client_secret": os.getenv("OAUTH_CLIENT_SECRET"),
                "code": code,
                "code_verifier": code_verifier,
                "grant_type": "authorization_code",
                "redirect_uri": os.getenv("OAUTH_REDIRECT_URI")
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code")
        token_data = response.json()
        access_token = jwt.encode(
            {"sub": token_data["id_token"], "exp": datetime.utcnow() + timedelta(hours=1)},
            os.getenv("OAUTH_CLIENT_SECRET"),
            algorithm="HS256"
        )
        return {"access_token": access_token, "token_type": "bearer"}
```

### server/webxos_wallet.py
```python
from server.security.crypto_engine import CryptoEngine
import re
import json
from typing import Dict
from pydantic import BaseModel, validator

class Wallet(BaseModel):
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
        self.wallets: Dict[str, Wallet] = {}

    def create_wallet(self, address: str, private_key: str, balance: float = 0.0) -> Wallet:
        wallet = Wallet(address=address, private_key=self.crypto.encrypt(private_key), balance=balance)
        self.wallets[address] = wallet
        return wallet

    def export_wallet(self, address: str, filename: str):
        if address not in self.wallets:
            raise ValueError("Wallet not found")
        wallet_data = self.wallets[address].dict()
        wallet_data['private_key'] = self.crypto.decrypt(wallet_data['private_key'])
        with open(filename, 'w') as f:
            f.write(json.dumps(wallet_data, indent=2))
        self.crypto.save_key(f"{filename}.key")

    def import_wallet(self, filename: str):
        with open(filename, 'r') as f:
            wallet_data = json.load(f)
        wallet = Wallet(**wallet_data)
        wallet.private_key = self.crypto.encrypt(wallet_data['private_key'])
        self.wallets[wallet.address] = wallet
        return wallet

    def sanitize_input(self, input_str: str) -> str:
        return re.sub(r'[<>{};]', '', input_str)
```

## Step 2: Validation
```bash
python -c "from server.security.crypto_engine import CryptoEngine; c = CryptoEngine('password'); enc = c.encrypt('test'); print(c.decrypt(enc))"
curl http://localhost:8000/mcp/auth/login
```

**Next**: Proceed to `part3.md` for API services.