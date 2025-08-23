import pytest
from sqlalchemy import create_engine, text
from server.services.database import SessionLocal
from pymongo import MongoClient
import os

@pytest.fixture
def sqlite_engine():
    return create_engine("sqlite:///:memory:")

@pytest.fixture
def postgres_engine():
    return create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg://user:password@localhost:5432/vial_mcp"))

@pytest.fixture
def mongo_client():
    return MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))

def test_sqlite_connection(sqlite_engine):
    with sqlite_engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).fetchone()
        assert result == (1,)

def test_postgres_connection(postgres_engine):
    with postgres_engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).fetchone()
        assert result == (1,)

def test_session_management():
    with SessionLocal() as session:
        result = session.execute(text("SELECT 1")).fetchone()
        assert result == (1,)
    assert session.is_active is False

def test_mongo_connection(mongo_client):
    client = mongo_client
    db = client["vial_mcp"]
    collection = db["test"]
    collection.insert_one({"test": "value"})
    result = collection.find_one({"test": "value"})
    assert result["test"] == "value"
