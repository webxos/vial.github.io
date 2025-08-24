from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
import redis
import pymongo
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter()

class HealthStatus(BaseModel):
    status: str
    services: dict

@router.get("/health")
async def health_check():
    """Check health of MCP services."""
    services = {
        "database": False,
        "redis": False,
        "mongodb": False
    }
    try:
        # Check SQLite
        engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL", "sqlite:///vial_mcp.db"))
        with engine.connect() as conn:
            services["database"] = True

        # Check Redis
        redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
        redis_client.ping()
        services["redis"] = True

        # Check MongoDB
        mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://mongodb:27017/vial_mcp"))
        mongo_client.server_info()
        services["mongodb"] = True

        logger.info("Health check passed")
        return HealthStatus(status="healthy", services=services)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthStatus(status="unhealthy", services=services)
