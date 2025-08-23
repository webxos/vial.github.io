import logging
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from pymongo import MongoClient
from httpx import AsyncClient

logger = logging.getLogger(__name__)
app = FastAPI()

class BIMMetadata(BaseModel):
    metadata: Dict[str, str]
    timestamp: str

@app.post("/v1/8bim/process")
async def process_8bim(image: UploadFile):
    """Process 8BIM metadata from image."""
    try:
        async with AsyncClient() as client:
            shield_response = await client.post(
                "https://api.azure.ai/content-safety/prompt-shields",
                json={"prompt": image.filename}
            )
            if shield_response.json().get("malicious"):
                raise HTTPException(status_code=400, detail="Malicious input detected")

        img_data = await image.read()
        img = Image.open(BytesIO(img_data))
        metadata = {}
        if hasattr(img, "_getexif"):
            exif_data = img._getexif() or {}
            for tag, value in exif_data.items():
                metadata[f"tag_{tag}"] = str(value)

        mongo_client = MongoClient("mongodb://mongo:27017")
        mongo_client.vial_mcp.research_data.insert_one({
            "type": "8BIM",
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        })
        return {"metadata": metadata}
    except Exception as e:
        logger.error(f"8BIM processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
