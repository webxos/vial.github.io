from fastapi import APIRouter, Depends, HTTPException
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])


@router.post("/register_tool")
async def register_tool(tool_data: dict, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        tool_id = str(uuid.uuid4())
        tool_info = {
            "tool_id": tool_id,
            "name": tool_data.get("name"),
            "type": tool_data.get("type", "langchain"),
            "endpoint": tool_data.get("endpoint", "/v1/mcp/execute")
        }
        await memory_manager.save_tool(tool_id, tool_info, request_id)
        logger.info(f"Registered tool {tool_info['name']}", request_id=request_id)
        return {"tool_id": tool_id, "status": "registered", "request_id": request_id}
    except Exception as e:
        logger.error(f"Tool registration error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute_tool")
async def execute_tool(tool_id: str, params: dict, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        tool = await memory_manager.get_tool(tool_id, request_id)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        result = {"status": "executed", "tool_id": tool_id, "params": params}
        logger.info(f"Executed tool {tool_id}", request_id=request_id)
        return {"result": result, "request_id": request_id}
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
