from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from server.config import settings
from server.logging import logger


engine = create_engine(settings.SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.log(f"Database session error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    try:
        from server.models.visual_components import VisualConfig
        from server.models.webxos_wallet import WalletModel
        from sqlalchemy import MetaData, Table, Column, String, JSON, DateTime
        metadata = MetaData()
        Table(
            "visual_configs",
            metadata,
            Column("id", String, primary_key=True),
            Column("name", String),
            Column("components", JSON),
            Column("connections", JSON),
            Column("metadata", JSON),
            Column("created_at", DateTime),
            Column("updated_at", DateTime)
        )
        Table(
            "wallets",
            metadata,
            Column("user_id", String, primary_key=True),
            Column("balance", String),
            Column("network_id", String),
            Column("created_at", DateTime),
            Column("updated_at", DateTime)
        )
        metadata.create_all(engine)
        logger.log("Database initialized: visual_configs, wallets tables created")
    except Exception as e:
        logger.log(f"Database initialization error: {str(e)}")
        raise
