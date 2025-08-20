from ..models.mcp_alchemist import mcp_alchemist
from ..models.webxos_wallet import webxos_wallet
from ..quantum_sync import quantum_sync
from ..security.oauth2 import create_access_token, get_current_user
from ..api.copilot_integration import copilot_client
import docker

class VialSDK:
    def __init__(self):
        self.docker_client = docker.from_client()
        self.alchemist = mcp_alchemist
        self.wallet = webxos_wallet
        self.quantum = quantum_sync

    async def initialize_system(self, user=None):
        if not user:
            return {"error": "Authentication required"}
        token = create_access_token({"sub": user["username"]})
        await self.quantum.broadcast_sync()
        wallet_id = self.wallet.get_wallet_status()["wallet_id"]
        return {"status": "initialized", "token": token, "wallet_id": wallet_id}

    async def train_and_sync(self, data: str, user=None):
        if not user:
            return {"error": "Authentication required"}
        train_result = await self.alchemist.train_wallet(data, user)
        if "error" not in train_result:
            await self.quantum.sync_node("node1", {"wallet_id": self.wallet.get_wallet_status()["wallet_id"], "data": data})
        return train_result

    def deploy_container(self, image_tag="vial_mcp_sdk:latest"):
        container = self.docker_client.containers.run(image_tag, detach=True, ports={'8000/tcp': 8000}, environment={"OAUTH_SECRET": "your-secret"})
        return {"container_id": container.id, "status": "deployed"}

vial_sdk = VialSDK()
