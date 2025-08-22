from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON, DateTime
from server.config import settings
from server.logging import logger


def run_migration():
    try:
        engine = create_engine(settings.SQLALCHEMY_DATABASE_URL)
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
        metadata.create_all(engine)
        logger.log("Database migration completed: visual_configs table created")
    except Exception as e:
        logger.log(f"Migration error: {str(e)}")
        raise


def rollback_migration():
    try:
        engine = create_engine(settings.SQLALCHEMY_DATABASE_URL)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        if "visual_configs" in metadata.tables:
            metadata.tables["visual_configs"].drop(engine)
        logger.log("Database rollback completed: visual_configs table dropped")
    except Exception as e:
        logger.log(f"Rollback error: {str(e)}")
        raise
