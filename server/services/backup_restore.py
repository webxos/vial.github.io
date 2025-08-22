from fastapi import FastAPI
from server.models.webxos_wallet import WalletModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from server.logging import logger

class BackupRestore:
    def __init__(self):
        self.engine = create_engine("sqlite:///vial.db")
        self.Session = sessionmaker(bind=self.engine)

    def backup_database(self, filename: str):
        with open(filename, "w") as f:
            session = self.Session()
            wallets = session.query(WalletModel).all()
            for wallet in wallets:
                f.write(f"{wallet.dict()}\n")
            session.close()
        logger.log(f"Backup created: {filename}")

def setup_backup_restore(app: FastAPI):
    backup = BackupRestore()
    app.state.backup_restore = backup

    @app.post("/backup")
    async def backup_endpoint(filename: str):
        app.state.backup_restore.backup_database(filename)
        return {"status": "backup completed"}
