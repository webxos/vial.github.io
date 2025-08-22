from pydantic import BaseModel
from typing import List, Dict, Any
from sqlalchemy import Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class VisualConfig(Base):
    __tablename__ = 'visual_configs'
    id = Column(String, primary_key=True)
    name = Column(String)
    components = Column(JSON)
    connections = Column(JSON)


class ComponentModel(BaseModel):
    id: str
    type: str
    title: str
    position: Dict[str, float]
    config: Dict[str, Any]
    connections: List[Dict]


class ConnectionModel(BaseModel):
    from_component: str
    to_component: str
    type: str
