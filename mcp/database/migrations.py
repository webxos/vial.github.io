from sqlalchemy import create_engine
from .data_models import Base
import os

engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///./vialmcp.db"))

def init_db():
    Base.metadata.create_all(bind=engine)
    return {"status": "database initialized", "time": "06:07 PM EDT, Aug 20, 2025"}
