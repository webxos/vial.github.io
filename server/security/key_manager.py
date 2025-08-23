import os
import logging
from typing import Optional
from pydantic import BaseModel
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class KeyPair(BaseModel):
    private_key: bytes
    public_key: bytes
    version: int
    created_at: datetime

class KeyManager:
    def __init__(self):
        self.key_store = {}

    def generate_key_pair(self, version: int = 1) -> KeyPair:
        """Generate RSA-4096 key pair."""
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(os.getenv("KEY_PASSWORD", "").encode())
        )
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return KeyPair(
            private_key=private_pem,
            public_key=public_key,
            version=version,
            created_at=datetime.utcnow()
        )

    def rotate_keys(self, key_id: str, rotation_days: int = 30) -> Optional[KeyPair]:
        """Rotate keys if expired."""
        current_key = self.key_store.get(key_id)
        if current_key and (datetime.utcnow() - current_key.created_at) > timedelta(days=rotation_days):
            new_key = self.generate_key_pair(version=current_key.version + 1)
            self.key_store[key_id] = new_key
            logger.info(f"Rotated key {key_id} to version {new_key.version}")
            return new_key
        return current_key
