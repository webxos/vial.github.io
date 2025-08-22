from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from server.services.database import get_db
from server.models.webxos_wallet import WalletModel
from server.logging import logger
import json
import os


def setup_backup_restore(app: FastAPI):
    @app.post("/backup")
    async def backup_data(db: Session = Depends(get_db)):
        try:
            wallets = db.query(WalletModel).all()
            backup_data = {
                "wallets": [
                    {
                        "user_id": w.user_id,
                        "balance": w.balance,
                        "network_id": w.network_id
                    } for w in wallets
                ],
                "timestamp": os.time()
            }
            backup_path = "data/backup.json"
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            with open(backup_path, "w") as f:
                json.dump(backup_data, f, indent=2)
            logger.log("Backup completed")
            return {"status": "backup_completed", "path": backup_path}
        except Exception as e:
            logger.log(f"Backup error: {str(e)}")
            return {"error": str(e)}

    @app.post("/restore")
    async def restore_data(db: Session = Depends(get_db)):
        try:
            backup_path = "data/backup.json"
            if not os.path.exists(backup_path):
                logger.log("Backup file not found")
                return {"error": "Backup file not found"}
            with open(backup_path, "r") as f:
                backup_data = json.load(f)
            for wallet_data in backup_data["wallets"]:
                wallet = WalletModel(**wallet_data)
                db.merge(wallet)
            db.commit()
            logger.log("Restore completed")
            return {"status": "restore_completed"}
        except Exception as e:
            logger.log(f"Restore error: {str(e)}")
            return {"error": str(e)}
