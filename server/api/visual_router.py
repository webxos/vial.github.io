from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from server.models.visual_components import ComponentModel, ConnectionModel
from server.services.database import get_db
from server.logging import logger
import uuid
import datetime
import json
import os


app = FastAPI()


class DiagramExportRequest(BaseModel):
    components: list[ComponentModel]
    connections: list[ConnectionModel]
    style: str = "default"


@app.get("/components/available")
async def get_available_components():
    request_id = str(uuid.uuid4())
    try:
        component_types = ["api_endpoint", "llm_model", "database", "tool", "agent"]
        logger.log("Fetched available components", request_id=request_id)
        return {"component_types": component_types}
    except Exception as e:
        logger.log(f"Error fetching components: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/components/validate")
async def validate_component(component: ComponentModel):
    request_id = str(uuid.uuid4())
    try:
        if component.type not in ["api_endpoint", "llm_model", "database", "tool", "agent"]:
            raise HTTPException(status_code=400, detail="Invalid component type")
        logger.log(f"Validated component: {component.id}", request_id=request_id)
        return {"status": "valid", "component_id": component.id}
    except Exception as e:
        logger.log(f"Component validation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/connections/create")
async def create_connection(connection: ConnectionModel, db=Depends(get_db)):
    request_id = str(uuid.uuid4())
    try:
        connection_id = str(uuid.uuid4())
        connection_data = {
            "id": connection_id,
            "from_component": connection.from_component,
            "to_component": connection.to_component,
            "type": connection.type,
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        with open(f"resources/connections/{connection_id}.json", "w") as f:
            json.dump(connection_data, f)
        logger.log(f"Created connection: {connection_id}", request_id=request_id)
        return {"status": "created", "connection_id": connection_id}
    except Exception as e:
        logger.log(f"Connection creation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagram/export")
async def export_diagram(request: DiagramExportRequest, db=Depends(get_db)):
    request_id = str(uuid.uuid4())
    try:
        fill_color = (
            "#3498db" if request.style == "default" else
            "#e74c3c" if request.style == "alert" else "#2ecc71"
        )
        svg_content = ['<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">']
        for component in request.components:
            x, y = component.position.x, component.position.y
            svg_content.append(
                f'<rect x="{x}" y="{y}" width="100" height="50" fill="{fill_color}" stroke="black"/>'
            )
            svg_content.append(
                f'<text x="{x + 10}" y="{y + 30}" fill="white">{component.title}</text>'
            )
        for connection in request.connections:
            from_comp = next(c for c in request.components if c.id == connection.from_component)
            to_comp = next(c for c in request.components if c.id == connection.to_component)
            svg_content.append(
                f'<line x1="{from_comp.position.x + 50}" y1="{from_comp.position.y + 50}" '
                f'x2="{to_comp.position.x + 50}" y2="{to_comp.position.y + 50}" '
                f'stroke="white" stroke-width="2"/>'
            )
        svg_content.append("</svg>")
        svg_path = f"resources/svg/diagram_{uuid.uuid4()}.svg"
        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
        with open(svg_path, "w") as f:
            f.write("\n".join(svg_content))
        logger.log(f"Exported SVG diagram to {svg_path}", request_id=request_id)
        return {"status": "exported", "svg_path": svg_path}
    except Exception as e:
        logger.log(f"Diagram export error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
