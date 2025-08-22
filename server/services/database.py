from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from server.models.webxos_wallet import Base as WalletBase
from server.models.visual_components import Base as VisualBase
from server.config.settings import settings


engine = create_engine(settings.sql_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()


def init_db():
    WalletBase.metadata.create_all(bind=engine)
    VisualBase.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
