import os
import json
import uuid
from datetime import datetime
from server.models.webxos_wallet import WalletModel
from server.models.visual_components import VisualConfig
from server.logging import logger
from sqlalchemy.orm import Session


class BackupRestoreService:
    def __init__(self):
        self.backup_dir = "resources/backups"

    async def backup_database(self, user_id: str, db: Session) -> dict:
        request_id = str(uuid.uuid4())
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            backup_path = (
                f"{self.backup_dir}/{user_id}_"
                f"{datetime.utcnow().isoformat()}.json"
            )
            wallets = db.query(WalletModel).filter(WalletModel.user_id == user_id).all()
            configs = db.query(VisualConfig).all()
            backup_data = {
                "wallets": [w.__dict__ for w in wallets],
                "configs": [c.__dict__ for c in configs],
                "timestamp": datetime.utcnow().isoformat()
            }
            with open(backup_path, "w") as f:
                json.dump(backup_data, f)
            logger.log(f"Database backup created at {backup_path}", request_id=request_id)
            return {"status": "backed_up", "backup_path": backup_path}
        except Exception as e:
            logger.log(f"Backup error: {str(e)}", request_id=request_id)
            return {"error": str(e)}

    async def restore_database(self, backup_path: str, db: Session) -> dict:
        request_id = str(uuid.uuid4())
        try:
            with open(backup_path, "r") as f:
                backup_data = json.load(f)
            db.query(WalletModel).delete()
            db.query(VisualConfig).delete()
            for wallet in backup_data["wallets"]:
                db.add(WalletModel(**wallet))
            for config in backup_data["configs"]:
                db.add(VisualConfig(**config))
            db.commit()
            logger.log(f"Database restored from {backup_path}", request_id=request_id)
            return {"status": "restored", "backup_path": backup_path}
        except Exception as e:
            logger.log(f"Restore error: {str(e)}", request_id=request_id)
            return {"error": str(e)}
