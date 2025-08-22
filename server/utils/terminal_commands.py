# server/utils/terminal_commands.py
import subprocess
from server.services.database import SessionLocal
from server.models.webxos_wallet import Wallet
import logging

logger = logging.getLogger(__name__)

def execute_terminal_command(command: str) -> dict:
    """Execute terminal command with wallet validation."""
    try:
        with SessionLocal() as session:
            wallet = session.query(Wallet).filter_by(
                address="test_wallet"
            ).first()
            if not wallet or wallet.reputation < 5.0:
                raise ValueError(
                    f"Insufficient reputation for command execution"
                )
        
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True
        )
        logger.info(
            f"Command executed: {command}, "
            f"output: {result.stdout}"
        )
        return {
            "status": "success",
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        logger.error(f"Command execution error: {str(e)}")
        raise
