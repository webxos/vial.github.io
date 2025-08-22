import requests
from server.services.advanced_logging import AdvancedLogger
from server.sdk.vial_sdk import VialSDK


logger = AdvancedLogger()


class VialSDKExtensions(VialSDK):
    def __init__(self, base_url: str, token: str):
        super().__init__(base_url, token)

    def export_diagram(self, config_id: str):
        response = requests.get(
            f"{self.base_url}/visual/diagram/export",
            params={"config_id": config_id},
            headers={"Authorization": f"Bearer {self.token}"}
        )
        self.logger.log("SDK diagram export attempted", extra={"config_id": config_id})
        return response.json()

    def trigger_backup(self):
        response = requests.post(
            f"{self.base_url}/backup",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        self.logger.log("SDK backup triggered", extra={"status": response.status_code})
        return response.json()
