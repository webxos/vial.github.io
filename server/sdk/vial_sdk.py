from server.services.advanced_logging import AdvancedLogger
import requests


logger = AdvancedLogger()


class VialSDK:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    def make_request(self, endpoint: str, data: dict):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(f"{self.base_url}/{endpoint}", json=data)
        logger.log("SDK request executed",
                   extra={"endpoint": endpoint,
                          "status_code": response.status_code})
        return response.json()
