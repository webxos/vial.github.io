from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.logging import logger
import json


class ComponentModel(BaseModel):
    id: str
    type: str
    title: str
    position: dict
    config: dict
    connections: list[dict]


class ConnectionModel(BaseModel):
    from_component: str
    to_component: str
    type: str


router = APIRouter()


@router.get("/components/available")
async def get_available_components():
    components = [
        {"type": "api_endpoint", "title": "API Endpoint"},
        {"type": "llm_model", "title": "LLM Model"},
        {"type": "database", "title": "Database"},
        {"type": "quantum_gate", "title": "Quantum Gate"}
    ]
    logger.log("Fetched available components", extra={"count": len(components)})
    return components


@router.post("/components/validate")
async def validate_component(component: ComponentModel, db: Session = Depends(get_db)):
    if not component.id or not component.type:
        logger.log("Invalid component configuration", extra={"component_id": component.id})
        return {"status": "error", "message": "Invalid component configuration"}
    logger.log("Component validated", extra={"component_id": component.id})
    return {"status": "valid", "component": component.dict()}


@router.post("/connections/create")
async def create_connection(connection: ConnectionModel, db: Session = Depends(get_db)):
    if not connection.from_component or not connection.to_component:
        logger.log("Invalid connection configuration", extra={"connection": connection.dict()})
        return {"status": "error", "message": "Invalid connection"}
    logger.log("Connection created", extra={"from": connection.from_component, "to": connection.to_component})
    return {"status": "created", "connection": connection.dict()}


@router.get("/diagram/export")
async def export_diagram(db: Session = Depends(get_db)):
    components = [
        {"id": f"comp{i}", "type": "api_endpoint", "position": {"x": i*50, "y": i*50}, "label": f"Comp {i}"}
        for i in range(1, 5)
    ]
    connections = [{"from_component": f"comp{i}", "to_component": f"comp{i+1}"} for i in range(1, 4)]
    svg = f"""<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
        {' '.join([f'<rect x="{c["position"]["x"]}" y="{c["position"]["y"]}" width="50" height="50" fill="green"/>' for c in components])}
        {' '.join([f'<line x1="{components[i]["position"]["x"]+25}" y1="{components[i]["position"]["y"]+25}" x2="{components[i+1]["position"]["x"]+25}" y2="{components[i+1]["position"]["y"]+25}" stroke="white"/>' for i in range(len(connections))])}
    </svg>"""
    logger.log("Exported SVG diagram", extra={"components": len(components)})
    return {"svg": svg}
