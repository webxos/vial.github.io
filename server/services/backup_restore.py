from fastapi import FastAPI
from server.models.webxos_wallet import WalletModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class BackupRestore:
    def __init__(self):
        self.engine = create_engine("sqlite:///wallet.db")
        self.Session = sessionmaker(bind=self.engine)

    def backup_database(self, filename: str):
        with open(filename, "w") as f:
            session = self.Session()
            wallets = session.query(WalletModel).all()
            for wallet in wallets:
                f.write(f"{wallet.dict()}\n")
            session.close()


def setup_backup_restore(app: FastAPI):
    backup = BackupRestore()
    app.state.backup_restore = backup

    @app.post("/backup")
    async def backup_endpoint(filename: str):
        app.state.backup_restore.backup_database(filename)
        return {"status": "backup completed"}
