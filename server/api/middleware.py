
async def logging_middleware(request, call_next):
    response = await call_next(request)
    print("Request logged")
    return response
