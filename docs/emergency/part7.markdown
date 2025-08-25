# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 7 (Database Models)

**Objective**: Implement SQLAlchemy database models with encryption for wallets and users.

**Instructions for LLM**:
1. Create `server/models/base.py` for database models.
2. Use SQLAlchemy with SQLite and integrate `CryptoEngine` for encryption.
3. Ensure compatibility with `server/webxos_wallet.py`.
4. Set up database initialization in `server/main.py`.

## Step 1: Create Database Models

### server/models/base.py
```python
from sqlalchemy import create_engine, Column, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from server.security.crypto_engine import CryptoEngine
import os

Base = declarative_base()
engine = create_engine("sqlite:///webxos.db")
Session = sessionmaker(bind=engine)

class Wallet(Base):
    __tablename__ = "wallets"
    id = Column(Integer, primary_key=True)
    address = Column(String, unique=True, nullable=False)
    private_key = Column(String, nullable=False)  # Encrypted
    balance = Column(Float, default=0.0)

    def __init__(self, address: str, private_key: str, balance: float = 0.0):
        crypto = CryptoEngine(os.getenv("WALLET_PASSWORD", "secure_wallet_password"))
        self.address = address
        self.private_key = crypto.encrypt(private_key)
        self.balance = balance

    def decrypt_private_key(self) -> str:
        crypto = CryptoEngine(os.getenv("WALLET_PASSWORD", "secure_wallet_password"))
        return crypto.decrypt(self.private_key)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    oauth_token = Column(String, nullable=False)  # Encrypted

    def __init__(self, email: str, oauth_token: str):
        crypto = CryptoEngine(os.getenv("WALLET_PASSWORD", "secure_wallet_password"))
        self.email = email
        self.oauth_token = crypto.encrypt(oauth_token)

    def decrypt_oauth_token(self) -> str:
        crypto = CryptoEngine(os.getenv("WALLET_PASSWORD", "secure_wallet_password"))
        return crypto.decrypt(self.oauth_token)

def init_db():
    Base.metadata.create_all(engine)
```

## Step 2: Update Main Application
Update `server/main.py` to initialize the database:
```python
from server.models.base import init_db
init_db()
```

## Step 3: Validation
```bash
python -c "from server.models.base import Wallet, Session; session = Session(); wallet = Wallet(address='0x1234567890abcdef1234567890abcdef12345678', private_key='a'*64); session.add(wallet); session.commit()"
sqlite3 webxos.db "SELECT * FROM wallets"
```

**Next**: Proceed to `part8.md` for monitoring.