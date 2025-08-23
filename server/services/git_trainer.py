from git import Repo
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid
import os

class GitTrainer:
    def __init__(self):
        self.repo = Repo(os.getcwd())
        self.alchemist = Alchemist()

    async def train_and_push(self, network_id: str, vial_id: str, commit_message: str) -> Dict:
        request_id = str(uuid.uuid4())
        try:
            await self.alchemist.train_vial({
                "vial_id": vial_id,
                "network_id": network_id
            }, request_id)
            self.repo.git.add(all=True)
            self.repo.git.commit(m=commit_message)
            self.repo.git.push()
            logger.info(
                f"Trained and pushed vial {vial_id} for {network_id}",
                request_id=request_id
            )
            return {"status": "success", "request_id": request_id}
        except Exception as e:
            logger.error(f"Train/push error: {str(e)}", request_id=request_id)
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-23T01:00:00Z]** Train/push error: {str(e)}\n")
            raise
