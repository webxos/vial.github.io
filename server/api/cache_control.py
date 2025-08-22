from fastapi import Request, Response
from fastapi.responses import JSONResponse

async def cache_response(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response
