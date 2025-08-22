# server/services/git_trainer.py
import httpx
import logging
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet

logger = logging.getLogger(__name__)

class GitTrainer:
    async def train_from_repo(self, repo_url: str, wallet_address: str):
        """Train model from Git repository with wallet validation."""
        try:
            with SessionLocal() as session:
                wallet = session.query(Wallet).filter_by(
                    address=wallet_address
                ).first()
                if not wallet:
                    raise ValueError(
                        f"Wallet not found for training: {wallet_address}"
                    )
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.github.com/repos/{repo_url}/contents",
                    headers={"Authorization": f"Bearer {wallet_address}"}
                )
                response.raise_for_status()
                content = response.json()
            logger.info(
                f"Training started from repo: {repo_url}, "
                f"Wallet: {wallet_address}"
            )
            return {"status": "success", "content": content}
        except Exception as e:
            logger.error(
                f"Training error for repo {repo_url}: {str(e)}"
            )
            raise
