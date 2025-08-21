from fastapi import Request
from server.services.audit_log import AuditLog
import time


async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    audit = AuditLog()
    response = await call_next(request)
    duration = time.time() - start_time
    await audit.log_action(
        action=f"{request.method} {request.url.path}",
        user_id="anonymous",
        details={"duration": duration, "status_code": response.status_code}
    )
    return response
