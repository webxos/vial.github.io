from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException
import os

Base = declarative_base()

class Star(Base):
    __tablename__ = "stars"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)

class DatabaseService:
    def __init__(self):
        db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/webxos")
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self):
        return self.SessionLocal()

    async def add_star(self, name: str, x: float, y: float, z: float):
        try:
            session = self.get_session()
            star = Star(name=name, x=x, y=y, z=z)
            session.add(star)
            session.commit()
            session.refresh(star)
            return star
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to add star: {str(e)}")
        finally:
            session.close()

    async def get_stars(self):
        try:
            session = self.get_session()
            stars = session.query(Star).all()
            return [{"id": star.id, "name": star.name, "x": star.x, "y": star.y, "z": star.z} for star in stars]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch stars: {str(e)}")
        finally:
            session.close()

database_service = DatabaseService()
