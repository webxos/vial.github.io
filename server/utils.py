import json
from server.models.webxos_wallet import WalletModel

def parse_json(data: str) -> WalletModel:
    parsed_data = json.loads(data)
    return WalletModel(
        user_id=parsed_data.get("user_id", ""),
        balance=parsed_data.get("balance", 0.0),
        network_id=parsed_data.get("network_id", "")
    )
