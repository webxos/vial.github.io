from server.security import verify_password
from server.logging import logger


class AuthAgent:
    def __init__(self):
        self.users = {"admin": "hashed_password"}


    def authenticate(self, username: str, password: str):
        stored_password = self.users.get(username)
        if stored_password and verify_password(password, stored_password):
            return {"status": "authenticated", "username": username}
        logger.error(f"Authentication failed for {username}")
        return {"status": "failed"}


auth_agent = AuthAgent()
