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

class AlibabaRequest(BaseModel):
    service: str  # e.g., "dataworks" or "opensearch"
    query: str

@app.post("/v1/alibaba/query", dependencies=[Depends(oauth2_scheme)])
async def query_alibaba(request: AlibabaRequest):
    """Query Alibaba Cloud service via MCP."""
    try:
        async with AsyncClient() as client:
            shield_response = await client.post(
                "https://api.azure.ai/content-safety/prompt-shields",
                json={"prompt": request.query}
            )
            if shield_response.json().get("malicious"):
                raise HTTPException(status_code=400, detail="Malicious input detected")

            # Placeholder for Alibaba Cloud API (use SDK in production)
            response = await client.get(
                f"https://api.alibabacloud.com/{request.service}",
                params={"query": request.query},
                headers={"Authorization": f"Bearer {os.getenv('ALIBABA_API_KEY')}"}
            )
            response.raise_for_status()
            data = response.json()

            mongo_client = MongoClient("mongodb://mongo:27017")
            mongo_client.vial_mcp.alibaba_data.insert_one({
                "type": "AlibabaCloud",
                "service": request.service,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            })
            return {"data": data}
    except Exception as e:
        logger.error(f"Alibaba Cloud query failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
