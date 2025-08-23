import os
import logging
from typing import Optional
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.argon2 import Argon2ID
from cryptography.exceptions import InvalidKey
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class EncryptionParams(BaseModel):
    data: bytes
    key: Optional[bytes] = None
    iv: Optional[bytes] = None

class CryptoEngine:
    def __init__(self):
        self.key_derivator = Argon2ID(salt=os.urandom(16), memory_cost=65536, time_cost=3, parallelism=4)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def encrypt(self, params: EncryptionParams) -> EncryptionParams:
        """Encrypt data with AES-256-GCM, derive key with Argon2id."""
        try:
            if not params.key:
                params.key = self.key_derivator.derive(os.getenv("ENCRYPTION_SECRET", "").encode())
            if not params.iv:
                params.iv = os.urandom(12)
            cipher = Cipher(algorithms.AES(params.key), modes.GCM(params.iv))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(params.data) + encryptor.finalize()
            return EncryptionParams(data=ciphertext, key=params.key, iv=params.iv)
        except InvalidKey as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt(self, params: EncryptionParams) -> bytes:
        """Decrypt AES-256-GCM encrypted data."""
        try:
            cipher = Cipher(algorithms.AES(params.key), modes.GCM(params.iv))
            decryptor = cipher.decryptor()
            return decryptor.update(params.data) + decryptor.finalize()
        except InvalidKey as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
