# server/security/security_headers.py
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)
        response.headers.update({
            "Content-Security-Policy": "default-src 'self';",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-Vercel-Protection": "enabled"
        })
        return response
