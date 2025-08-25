from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64

class CryptoEngine:
    def __init__(self, password: str = None):
        # Derive key from password or generate random key
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
        self.iv = os.urandom(12)  # GCM recommends 12-byte IV

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
