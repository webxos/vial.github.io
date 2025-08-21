from fastapi import APIRouter
from server.services.redis_handler import test_redis_connection
from server.services.mongodb_handler import MongoDBHandler


router = APIRouter()


@router.get("/health/full")
async def full_health_check():
    redis_status = await test_redis_connection()
    mongo = MongoDBHandler()
    mongo_status = await mongo.get_metadata({"health": "test"}) or {"status": "connected"}
    return {
        "redis": redis_status,
        "mongo": mongo_status,
        "api": {"status": "ok"}
    }
