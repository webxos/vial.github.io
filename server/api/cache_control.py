from fastapi import Request, Response
from server.services.redis_handler import redis_client
import hashlib
import json

async def cache_response(request: Request, call_next):
    cache_key = f"cache:{hashlib.md5(f'{request.method}:{request.url}'.encode()).hexdigest()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return Response(content=cached, media_type="application/json")
    response = await call_next(request)
    if response.status_code == 200:
        await redis_client.setex(cache_key, 3600, response.body)
    return response
