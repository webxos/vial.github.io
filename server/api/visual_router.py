from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import uuid
import json
from datetime import datetime
from server.services.mcp_alchemist import Alchemist
from server.mcp.auth import oauth2_scheme
from server.logging import logger

router = APIRouter()


class Component(BaseModel):
    id: str
    type: str
    title: str
    position: dict


class Connection(BaseModel):
    source_id: str
    target_id: str
    type: str


class Diagram(BaseModel):
    components: list[Component]
    connections: list[Connection]


@router.post("/visual/diagram/export")
async def export_diagram(diagram: Diagram, token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        if not any(component.type not in ["api_endpoint", "quantum", "deploy"] for component in diagram.components):
            raise HTTPException(status_code=400, detail="Invalid component type")
        
        svg_content = "<svg>...</svg>"  # Placeholder for SVG generation
        logger.log(f"Diagram exported: {request_id}", request_id=request_id)
        return {"svg_content": svg_content, "request_id": request_id}
    except Exception as e:
        logger.log(f"Diagram export error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visual/diagram/task")
async def process_diagram_task(task: str, diagram: Diagram, token: str = Depends(oauth2_scheme), alchemist: Alchemist = Depends(Alchemist)):
    request_id = str(uuid.uuid4())
    try:
        from server.mcp.auth import map_oauth_to_mcp_session
        await map_oauth_to_mcp_session(token, request_id)
        context = {
            "diagram": json.loads(diagram.json()),
            "timestamp": datetime.now().isoformat()
        }
        result = await alchemist.delegate_task(task, context)
        logger.log(f"Diagram task processed: {task}", request_id=request_id)
        return {"result": result, "request_id": request_id}
    except Exception as e:
        logger.log(f"Diagram task error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
