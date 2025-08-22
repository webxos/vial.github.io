from fastapi import FastAPI
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.models.visual_components import VisualConfig
from server.services.advanced_logging import AdvancedLogger
import json


def setup_backup_restore(app: FastAPI):
    logger = AdvancedLogger()

    async def backup_configs(db: Session = Depends(get_db)):
        configs = db.query(VisualConfig).all()
        backup_data = [{"id": c.id, "name": c.name, "components": c.components, "connections": c.connections} for c in configs]
        with open("/app/data/backup.json", "w") as f:
            json.dump(backup_data, f)
        logger.log("Configuration backup created", extra={"count": len(backup_data)})
        return {"status": "backed up", "count": len(backup_data)}

    async def restore_configs(db: Session = Depends(get_db)):
        try:
            with open("/app/data/backup.json", "r") as f:
                backup_data = json.load(f)
            for config in backup_data:
                db.merge(VisualConfig(**config))
            db.commit()
            logger.log("Configuration restored", extra={"count": len(backup_data)})
            return {"status": "restored", "count": len(backup_data)}
        except Exception as e:
            logger.log("Restore failed", extra={"error": str(e)})
            return {"error": str(e)}

    app.state.backup_configs = backup_configs
    app.state.restore_configs = restore_configs
    logger.log("Backup/restore initialized", extra={"system": "backup_restore"})
