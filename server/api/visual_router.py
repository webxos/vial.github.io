from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.models.visual_components import VisualConfig
from server.services.advanced_logging import AdvancedLogger


router = APIRouter()
logger = AdvancedLogger()


@router.get("/visual/diagram/export")
async def export_diagram(config_id: str, db: Session = Depends(get_db)):
    config = db.query(VisualConfig).filter(VisualConfig.id == config_id).first()
    if not config:
        logger.log("Diagram export failed: Config not found", extra={"config_id": config_id})
        return {"error": "Config not found"}
    svg_data = {"id": config.id, "components": config.components, "connections": config.connections}
    logger.log("Diagram exported", extra={"config_id": config_id})
    return {"status": "exported", "svg_data": svg_data}
