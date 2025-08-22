from pydantic import BaseModel
from typing import Dict, Any, List
from enum import Enum


class ComponentType(str, Enum):
    API_ENDPOINT = "api_endpoint"
    LLM_MODEL = "llm_model"
    DATABASE = "database"
    TOOL = "tool"
    AGENT = "agent"


class ConnectionType(str, Enum):
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"


class Position3D(BaseModel):
    x: float
    y: float
    z: float


class ComponentModel(BaseModel):
    id: str
    type: ComponentType
    title: str
    position: Position3D
    config: Dict[str, Any]
    connections: List[str]


class ConnectionModel(BaseModel):
    from_component: str
    to_component: str
    type: ConnectionType
