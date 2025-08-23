from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
from fastapi.security import OAuth2PasswordBearer
import uuid

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

class DataSchema(BaseModel):
    id: str
    content: str
    network_id: str

@router.post("/data")
async def create_data(data: DataSchema, token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        alchemist = Alchemist()
        result = await alchemist.perform_crud({
            "operation": "create",
            "data": data.dict()
        }, request_id)
        logger.info(f"Created data for {data.network_id}", request_id=request_id)
        return result
    except Exception as e:
        logger.error(f"Create data error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/data/{network_id}")
async def read_data(network_id: str, token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        alchemist = Alchemist()
        result = await alchemist.perform_crud({
            "operation": "read",
            "data": {"network_id": network_id}
        }, request_id)
        logger.info(f"Read data for {network_id}", request_id=request_id)
        return result
    except Exception as e:
        logger.error(f"Read data error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=400, detail=str(e))
