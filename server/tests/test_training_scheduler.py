import json
from fastapi import HTTPException


def parse_json(data: str):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")


def validate_wallet(data: dict):
    required_fields = ["user_id", "balance", "network_id"]
    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required wallet fields")
    return data
