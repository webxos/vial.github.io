import httpx
from server.config import settings


class VialSDK:
    def __init__(self, api_base_url: str = settings.API_BASE_URL):
        self.client = httpx.AsyncClient(base_url=api_base_url)
        self.token = None

    async def authenticate(self, username: str, password: str):
        response = await self.client.post(
            "/auth/token",
            json={"username": username, "password": password}
        )
        response.raise_for_status()
        self.token = response.json().get("access_token")
        return self.token

    async def generate_credentials(self):
        response = await self.client.post(
            "/auth/generate-credentials",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        response.raise_for_status()
        return response.json()

    async def execute_quantum_circuit(self, circuit: dict, backend: str = "qasm_simulator"):
        response = await self.client.post(
            "/quantum/execute",
            json={"circuit": circuit, "backend": backend},
            headers={"Authorization": f"Bearer {self.token}"}
        )
        response.raise_for_status()
        return response.json()
