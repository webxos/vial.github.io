import pytest
from server.utils import parse_jsonrpc_request, build_jsonrpc_response
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_parse_jsonrpc_request():
    valid_request = {"jsonrpc": "2.0", "method": "test_method", "params": {}, "id": "1"}
    method, params, id = parse_jsonrpc_request(valid_request)
    assert method == "test_method"
    assert params == {}
    assert id == "1"

    invalid_request = {"jsonrpc": "1.0", "method": "test_method", "id": "1"}
    with pytest.raises(HTTPException) as exc:
        parse_jsonrpc_request(invalid_request)
    assert exc.value.status_code == 400
    assert "JSON-RPC version must be 2.0" in str(exc.value)


@pytest.mark.asyncio
async def test_build_jsonrpc_response():
    response = build_jsonrpc_response(id="1", result={"status": "ok"})
    assert response == {"jsonrpc": "2.0", "id": "1", "result": {"status": "ok"}}

    error_response = build_jsonrpc_response(id="2", error={"code": -32601, "message": "Method not found"})
    assert error_response == {"jsonrpc": "2.0", "id": "2", "error": {"code": -32601, "message": "Method not found"}}
