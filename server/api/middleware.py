```python
from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader
import os

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def oauth_middleware(request: Request):
    api_key = await api_key_header(request)
    expected_key = os.getenv("NASA_API_KEY")
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing NASA API key")
    return request
```
