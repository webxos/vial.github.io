import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from httpx import AsyncClient
from pymongo import MongoClient
from datetime import datetime

logger = logging.getLogger(__name__)
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")

class ServiceNowRequest(BaseModel):
    table: str
    query: str

@app.post("/v1/servicenow/query", dependencies=[Depends(oauth2_scheme)])
async def query_servicenow(request: ServiceNowRequest):
    """Query ServiceNow table via MCP."""
    try:
        async with AsyncClient() as client:
            shield_response = await client.post(
                "https://api.azure.ai/content-safety/prompt-shields",
                json={"prompt": request.query}
            )
            if shield_response.json().get("malicious"):
                raise HTTPException(status_code=400, detail="Malicious input detected")

            response = await client.get(
                f"https://your-instance.service-now.com/api/now/table/{request.table}",
                params={"sysparm_query": request.query},
                auth=("your_username", "your_password")  # Use OAuth in production
            )
            response.raise_for_status()
            data = response.json()

            mongo_client = MongoClient("mongodb://mongo:27017")
            mongo_client.vial_mcp.servicenow_data.insert_one({
                "type": "ServiceNow",
                "table": request.table,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            })
            return {"data": data}
    except Exception as e:
        logger.error(f"ServiceNow query failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
