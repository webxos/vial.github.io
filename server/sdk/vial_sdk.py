from server.logging import logger
import requests


class VialSDK:
    def __init__(self):
        self.api_key = "default_key"
        self.base_url = "http://localhost:8000"

    def execute(self, command: dict):
        try:
            response = requests.post(f"{self.base_url}/jsonrpc", json=command)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"SDK execution failed: {str(e)}")
            return {"status": "failed", "error": str(e)}


vial_sdk = VialSDK()
