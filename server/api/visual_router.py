from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from server.services.database import get_db
from server.models.visual_components import ComponentModel, ConnectionModel
from server.logging import logger

router = APIRouter()


class VisualConfig(BaseModel):
    components: List[ComponentModel]
    connections: List[ConnectionModel]


@router.get("/components/available")
async def get_available_components():
    return {
        "components": [
            {"type": "api_endpoint", "title": "API Endpoint"},
            {"type": "llm_model", "title": "LLM Model"},
            {"type": "database", "title": "Database"},
            {"type": "tool", "title": "Tool"},
            {"type": "agent", "title": "Agent"}
        ]
    }


@router.post("/components/validate")
async def validate_component(component: ComponentModel):
    try:
        if not component.type in [c["type"] for c in (await get_available_components())["components"]]:
            raise ValueError(f"Invalid component type: {component.type}")
        logger.log(f"Validated component: {component.id}")
        return {"status": "valid", "component": component}
    except Exception as e:
        logger.log(f"Component validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/connections/create")
async def create_connection(connection: ConnectionModel):
    try:
        logger.log(f"Created connection: {connection.from_component} -> {connection.to_component}")
        return {"status": "created", "connection": connection}
    except Exception as e:
        logger.log(f"Connection creation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/save-config")
async def save_config(config: VisualConfig, db=Depends(get_db)):
    try:
        config_data = config.dict()
        db.execute(
            "INSERT INTO visual_configs (id, name, components, connections, created_at, updated_at) "
            "VALUES (:id, :name, :components, :connections, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            {
                "id": str(uuid.uuid4()),
                "name": "config_" + datetime.utcnow().isoformat(),
                "components": json.dumps([c.dict() for c in config.components]),
                "connections": json.dumps([c.dict() for c in config.connections])
            }
        )
        db.commit()
        logger.log("Saved visual configuration")
        return {"status": "saved", "config_id": config_data["id"]}
    except Exception as e:
        logger.log(f"Config save error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagram/export")
async def export_diagram(config_id: str, db=Depends(get_db)):
    try:
        config = db.execute(
            "SELECT components, connections FROM visual_configs WHERE id = :id",
            {"id": config_id}
        ).fetchone()
        if not config:
            raise ValueError("Config not found")
        components = json.loads(config["components"])
        svg = f"""<svg width="400" height="400">
            <rect x="0" y="0" width="400" height="400" fill="#f0f0f0"/>
            {"".join(f'<rect x="{c["position"]["x"]}" y="{c["position"]["y"]}" width="50" height="50" fill="#3498db"/><text x="{c["position"]["x"]+10}" y="{c["position"]["y"]+30}" fill="white">{c["title"]}</text>' for c in components)}
        </svg>"""
        logger.log(f"Exported SVG diagram for config: {config_id}")
        return {"svg": svg}
    except Exception as e:
        logger.log(f"Diagram export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
