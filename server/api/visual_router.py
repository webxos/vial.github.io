from fastapi import APIRouter, Depends, HTTPException
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid
import json

router = APIRouter(prefix="/v1/visual", tags=["visual"])


@router.post("/generate_workflow")
async def generate_workflow(workflow_data: dict, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        nodes = workflow_data.get("nodes", [])
        edges = workflow_data.get("edges", [])
        svg_content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400">\n'
        )
        for node in nodes:
            x, y = node.get("x", 0), node.get("y", 0)
            svg_content += (
                f'<rect x="{x}" y="{y}" width="100" height="50" fill="#4CAF50" stroke="black"/>\n'
                f'<text x="{x + 50}" y="{y + 30}" text-anchor="middle">{node.get("label", "Node")}</text>\n'
            )
        for edge in edges:
            start = edge.get("start", {"x": 0, "y": 0})
            end = edge.get("end", {"x": 100, "y": 100})
            svg_content += (
                f'<line x1="{start["x"]}" y1="{start["y"]}" x2="{end["x"]}" y2="{end["y"]}" '
                'stroke="black" stroke-width="2"/>\n'
            )
        svg_content += '</svg>'
        await memory_manager.save_workflow(str(uuid.uuid4()), {"svg": svg_content, "nodes": nodes, "edges": edges}, request_id)
        logger.info(f"Generated SVG workflow", request_id=request_id)
        return {"svg": svg_content, "request_id": request_id}
    except Exception as e:
        logger.error(f"Workflow generation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
