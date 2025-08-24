import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from server.api.error_handler import global_exception_handler
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_global_exception_handler_http_exception():
    request = AsyncMock(spec=Request)
    request.url.path = "/mcp/test"
    request.scope = {"db": None, "wallet_id": "wallet123"}
    exc = HTTPException(status_code=400, detail="Bad request")
    
    response = await global_exception_handler(request, exc)
    
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert response.body.decode() == '{"detail":"Bad request"}'

@pytest.mark.asyncio
async def test_global_exception_handler_generic_exception():
    request = AsyncMock(spec=Request)
    request.url.path = "/mcp/test"
    request.scope = {"db": None, "wallet_id": "wallet123"}
    exc = ValueError("Test error")
    
    response = await global_exception_handler(request, exc)
    
    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert "Internal server error: Test error" in response.body.decode()
