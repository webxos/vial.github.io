from fastapi import Request, Response
from fastapi.responses import JSONResponse

async def rate_limit(request: Request, call_next):
    # Simple in-memory rate limiting (replace with Redis for production)
    request.state.rate_limit_count = getattr(request.state, "rate_limit_count", 0) + 1
    if request.state.rate_limit_count > 100:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    response: Response = await call_next(request)
    return response
