from sqlalchemy import Column, String, Integer, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from server.logging import logger
import uuid


Base = declarative_base()


class SwarmAgentModel(Base):
    __tablename__ = "swarm_agents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    instructions = Column(String, nullable=False)
    functions = Column(JSON, default=[])
    active = Column(Boolean, default=True)
    created_at = Column(Integer, default=lambda: int(__import__('time').time()))

    def __repr__(self):
        return f"<SwarmAgentModel(name={self.name})>"

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "instructions": self.instructions,
            "functions": self.functions,
            "active": self.active,
            "created_at": self.created_at
        }
