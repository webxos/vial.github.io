# server/services/agent_tasks.py
from server.services.vial_manager import VialManager
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AgentTasks:
    def __init__(self):
        self.vial_manager = VialManager()

    async def execute_task(self, task_id: str, wallet_address: str) -> Dict[str, Any]:
        """Execute agent task with reputation check."""
        try:
            with SessionLocal() as session:
                wallet = session.query(Wallet).filter_by(
                    address=wallet_address
                ).first()
                if not wallet or wallet.reputation < 10.0:
                    raise ValueError(
                        f"Insufficient reputation for {wallet_address}"
                    )
                result = await self.vial_manager.train_vial(task_id)
                logger.info(
                    f"Task {task_id} executed for wallet {wallet_address} "
                    f"with reputation {wallet.reputation}"
                )
                return result
        except Exception as e:
            logger.error(f"Task execution error: {str(e)}")
            raise
