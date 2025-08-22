from fastapi import Request, HTTPException
from server.services.advanced_logging import AdvancedLogger
import time


logger = AdvancedLogger()
requests = {}


async def rate_limit(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    if client_ip not in requests:
        requests[client_ip] = []
    
    requests[client_ip] = [t for t in requests[client_ip] if current_time - t < 60]
    if len(requests[client_ip]) >= 100:
        logger.log("Rate limit exceeded", extra={"client_ip": client_ip})
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    requests[client_ip].append(current_time)
    logger.log("Rate limit check passed", extra={"client_ip": client_ip})
    return await call_next(request)
