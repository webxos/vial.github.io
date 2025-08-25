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
