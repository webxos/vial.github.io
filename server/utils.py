import json


def parse_json(data):
    return json.loads(data)


def validate_wallet(data):
    return {"user_id": data.get("user_id"), "balance": data.get("balance")}
