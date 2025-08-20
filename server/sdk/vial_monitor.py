from ..models.mcp_alchemist import mcp_alchemist
from ..models.webxos_wallet import webxos_wallet
from ..quantum_sync import quantum_sync
from ..logging import logger
import psutil
import docker

class VialMonitor:
    def __init__(self):
        self.docker_client = docker.from_client()
        self.alchemist = mcp_alchemist
        self.wallet = webxos_wallet
        self.quantum = quantum_sync

    def check_system_health(self):
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        containers = self.docker_client.containers.list()
        wallet_status = self.wallet.get_wallet_status()
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory.percent,
            "container_count": len(containers),
            "wallet_status": wallet_status,
            "quantum_sync": self.quantum.network_nodes
        }

    async def monitor_training(self):
        result = await self.alchemist.train_wallet("1,2,3,4,5", {"username": "system"})
        logger.info(f"Training monitor result: {result}")
        return result

vial_monitor = VialMonitor()
