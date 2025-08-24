from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from server.models.audit_repository import AuditRepository
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, filename="logs/api_errors.log")

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle global exceptions and log to audit repository."""
    try:
        db: Session = request.scope.get("db")
        audit_repo = AuditRepository(db)
        wallet_id = request.scope.get("wallet_id", "unknown")
        audit_repo.log_action(
            action="error",
            wallet_id=wallet_id,
            endpoint=str(request.url.path),
            details=str(exc)
        )
    except Exception as e:
        logging.error(f"Audit logging failed in error handler: {str(e)}")
    
    status_code = 500
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        detail = exc.detail
    else:
        detail = f"Internal server error: {str(exc)}"
    
    logging.error(f"API error on {request.url.path}: {detail}")
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail}
    )
