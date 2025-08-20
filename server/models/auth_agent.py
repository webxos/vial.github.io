from server.security import verify_token, create_access_token
from server.services.database import User, sessionmaker, create_engine
from server.config import get_settings
from fastapi import HTTPException

class AuthAgent:
    def __init__(self):
        self.settings = get_settings()
        self.engine = create_engine(self.settings.DATABASE_URL, connect_args={"check_same_thread": False})
        self.Session = sessionmaker(bind=self.engine)

    def authenticate_user(self, username: str, password: str):
        session = self.Session()
        try:
            user = session.query(User).filter_by(username=username).first()
            if not user or not verify_password(password, user.hashed_password):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            token = create_access_token({"sub": username})
            return {"access_token": token, "token_type": "bearer"}
        finally:
            session.close()

    def validate_session(self, token: str):
        return verify_token(token)

auth_agent = AuthAgent()
