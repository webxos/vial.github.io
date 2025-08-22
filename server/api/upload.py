from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.services.advanced_logging import AdvancedLogger
from server.models.visual_components import VisualConfig
import json


router = APIRouter()
logger = AdvancedLogger()


@router.post("/upload-config")
async def upload_config(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        content = await file.read()
        config_data = json.loads(content.decode('utf-8'))
        db_config = VisualConfig(
            id=config_data.get("id"),
            name=config_data.get("name"),
            components=config_data.get("components"),
            connections=config_data.get("connections")
        )
        db.add(db_config)
        db.commit()
        logger.log("Configuration uploaded", extra={"config_id": db_config.id})
        return {"status": "uploaded", "config_id": db_config.id}
    except Exception as e:
        logger.log("Configuration upload failed", extra={"error": str(e)})
        return {"error": str(e)}
