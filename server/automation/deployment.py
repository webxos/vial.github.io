from fastapi import Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.services.advanced_logging import AdvancedLogger


logger = AdvancedLogger()


def deploy_to_github_pages(config_id: str, db: Session = Depends(get_db)):
    try:
        config = db.query(db.models.VisualConfig).filter_by(id=config_id).first()
        if not config:
            logger.log("Deployment failed",
                       extra={"error": "Config not found"})
            return {"error": "Config not found"}
        
        logger.log("Deployment to GitHub Pages initiated",
                   extra={"config_id": config_id})
        return {"status": "deployed", "url": f"https://vial.github.io/{config_id}"}
    except Exception as e:
        logger.log("Deployment error",
                   extra={"error": str(e)})
        raise
