from fastapi import Request
from fastapi.responses import JSONResponse


async def rate_limit(request: Request, call_next):
    return await call_next(request)
