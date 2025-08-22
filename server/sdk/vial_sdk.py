import requests
from server.services.advanced_logging import AdvancedLogger


class VialSDK:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
        self.logger = AdvancedLogger()

    def save_config(self, config: dict):
        response = requests.post(
            f"{self.base_url}/save-config",
            json=config,
            headers={"Authorization": f"Bearer {self.token}"}
        )
        self.logger.log("SDK config save attempted", extra={"config_name": config.get("name")})
        return response.json()

    def train_vial(self, vial_id: str):
        response = requests.post(
            f"{self.base_url}/agent/train",
            json={"vial_id": vial_id},
            headers={"Authorization": f"Bearer {self.token}"}
        )
        self.logger.log("SDK vial training attempted", extra={"vial_id": vial_id})
        return response.json()
