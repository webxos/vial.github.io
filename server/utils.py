import json
from fastapi import HTTPException


def parse_jsonrpc_request(data: dict):
    if not all(key in data for key in ["jsonrpc", "method", "id"]):
        raise HTTPException(status_code=400, detail="Invalid JSON-RPC request")
    if data["jsonrpc"] != "2.0":
        raise HTTPException(status_code=400, detail="JSON-RPC version must be 2.0")
    return data["method"], data.get("params", {}), data["id"]


def build_jsonrpc_response(id: str, result: dict = None, error: dict = None):
    response = {"jsonrpc": "2.0", "id": id}
    if result is not None:
        response["result"] = result
    if error is not None:
        response["error"] = error
    return response
