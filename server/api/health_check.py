from fastapi import APIRouter, Depends
from server.security import verify_token
from server.logging import logger
from pymongo import MongoClient
from redis import Redis
from sqlalchemy import create_engine
from server.config import get_settings

router = APIRouter(prefix="/jsonrpc", tags=["health"])
settings = get_settings()

@router.post("/health_check")
async def advanced_health_check(token: str = Depends(verify_token)):
    health_status = {"status": "healthy", "services": {}}
    
    # Check MongoDB
    try:
        client = MongoClient(settings.MONGO_URL, serverSelectionTimeoutMS=5000)
        client.server_info()
        health_status["services"]["mongodb"] = "healthy"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        health_status["services"]["mongodb"] = "unhealthy"
    
    # Check Redis
    try:
        redis = Redis.from_url(settings.REDIS_URL)
        redis.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        health_status["services"]["redis"] = "unhealthy"
    
    # Check SQLite
    try:
        engine = create_engine(settings.DATABASE_URL)
        engine.connect()
        health_status["services"]["sqlite"] = "healthy"
    except Exception as e:
        logger.error(f"SQLite health check failed: {str(e)}")
        health_status["services"]["sqlite"] = "unhealthy"
    
    return {"jsonrpc": "2.0", "result": health_status, "id": 1}
