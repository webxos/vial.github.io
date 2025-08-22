from fastapi import Request
from server.api.security import SecurityManager


class SecurityHeaders:
    def __init__(self):
        self.security = SecurityManager()

    async def add_headers(self, request: Request, call_next):
        response = await call_next(request)
        headers = self.security.get_security_headers()
        for key, value in headers.items():
            response.headers[key] = value
        return response


security_headers = SecurityHeaders()
