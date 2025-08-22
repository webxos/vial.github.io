from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.models.visual_components import VisualConfig
from server.services.advanced_logging import AdvancedLogger


router = APIRouter()
logger = AdvancedLogger()


@router.get("/export/json/{config_id}")
async def export_json(config_id: str, db: Session = Depends(get_db)):
    config = db.query(VisualConfig).filter(VisualConfig.id == config_id).first()
    if not config:
        logger.log("JSON export failed",
                   extra={"config_id": config_id})
        return {"error": "Config not found"}
    json_data = {
        "id": config.id,
        "name": config.name,
        "components": config.components,
        "connections": config.connections
    }
    logger.log("JSON exported",
               extra={"config_id": config_id})
    return {"status": "exported", "json_data": json_data}
