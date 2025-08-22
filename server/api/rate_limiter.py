
async def rate_limit(request, call_next):
    response = await call_next(request)
    response.headers["X-Rate-Limit"] = "10"
    return response
