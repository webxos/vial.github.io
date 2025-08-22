# server/api/visual_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from server.services.database import SessionLocal
from server.models.visual_components import ComponentModel, ConnectionModel
import logging
from typing import List, Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

class VisualConfigModel(BaseModel):
    components: List[ComponentModel]
    connections: List[ConnectionModel]

@router.post("/save-config")
async def save_config(config: VisualConfigModel):
    """Save visual API router configuration."""
    try:
        with SessionLocal() as session:
            for component in config.components:
                session.add(component)
            for connection in config.connections:
                session.add(connection)
            session.commit()
            logger.info("Configuration saved successfully")
            return {"status": "success"}
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/diagram/export")
async def export_diagram():
    """Export visual configuration as SVG."""
    with SessionLocal() as session:
        components = session.query(ComponentModel).all()
        connections = session.query(ConnectionModel).all()
        svg_content = (
            '<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">'
        )
        for comp in components:
            svg_content += (
                f'<rect x="{comp.position.x}" y="{comp.position.y}" '
                f'width="100" height="50" fill="green" stroke="black"/>'
                f'<text x="{comp.position.x + 10}" y="{comp.position.y + 30}" '
                f'fill="white">{comp.title}</text>'
            )
        for conn in connections:
            svg_content += (
                f'<line x1="{conn.from_component}" y1="100" '
                f'x2="{conn.to_component}" y2="150" stroke="white" '
                f'stroke-width="2"/>'
            )
        svg_content += '</svg>'
        return Response(content=svg_content, media_type="image/svg+xml")
