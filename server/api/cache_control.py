
async def cache_response(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response
