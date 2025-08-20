from server.services.database import init_db
from server.models.webxos_wallet import webxos_wallet
from server.config import get_settings
import docker

class VialSDK:
    def __init__(self):
        self.settings = get_settings()
        self.docker_client = docker.from_env()

    def initialize_system(self, config: dict):
        # Initialize database
        init_db()
        
        # Create default wallet for user
        username = config.get("username", "system")
        wallet = webxos_wallet.create_wallet(username)
        
        # Start Docker containers
        try:
            self.docker_client.containers.run(
                "vial_app",
                detach=True,
                environment={
                    "OAUTH_SECRET": self.settings.OAUTH_SECRET,
                    "JWT_SECRET": self.settings.JWT_SECRET,
                    "MONGO_URL": self.settings.MONGO_URL,
                    "REDIS_URL": self.settings.REDIS_URL,
                    "DATABASE_URL": self.settings.DATABASE_URL
                }
            )
        except docker.errors.DockerException as e:
            raise ValueError(f"Failed to start container: {str(e)}")
        
        return {"status": "system initialized", "wallet_address": wallet["address"]}

vial_sdk = VialSDK()
