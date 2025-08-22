from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base
from server.services.advanced_logging import AdvancedLogger


Base = declarative_base()
logger = AdvancedLogger()


class AuthAgent(Base):
    __tablename__ = "auth_agents"
    id = Column(String, primary_key=True)
    username = Column(String, unique=True)
    role = Column(String, default="user")

    def __init__(self, id: str, username: str, role: str = "user"):
        self.id = id
        self.username = username
        self.role = role
        logger.log("Auth agent created", extra={"username": username, "role": role})
