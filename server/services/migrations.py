from alembic import command
from alembic.config import Config
from server.services.database import get_db
from server.models.webxos_wallet import Wallet
from server.models.auth_agent import User
from sqlalchemy import create_engine
import os

def init_migrations():
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", os.getenv("SQLITE_DB_PATH", "sqlite:///vial/database.sqlite"))
    command.upgrade(alembic_cfg, "head")

async def create_migration(revision_message: str):
    alembic_cfg = Config("alembic.ini")
    async with get_db() as db:
        engine = create_engine(os.getenv("SQLITE_DB_PATH", "sqlite:///vial/database.sqlite"))
        db.bind = engine
        command.revision(alembic_cfg, message=revision_message, autogenerate=True)

async def apply_migrations():
    alembic_cfg = Config("alembic.ini")
    async with get_db() as db:
        engine = create_engine(os.getenv("SQLITE_DB_PATH", "sqlite:///vial/database.sqlite"))
        db.bind = engine
        command.upgrade(alembic_cfg, "head")

def setup_database_schema():
    async with get_db() as db:
        engine = create_engine(os.getenv("SQLITE_DB_PATH", "sqlite:///vial/database.sqlite"))
        Wallet.metadata.create_all(engine)
        User.metadata.create_all(engine)
