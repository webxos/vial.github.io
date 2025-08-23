from server.models.webxos_wallet import Wallet
from server.services.database import SessionLocal
from server.logging import logger
import subprocess
import uuid

def execute_terminal_command(command: str) -> dict:
    request_id = str(uuid.uuid4())
    try:
        with SessionLocal() as db:
            wallet = db.query(Wallet).filter(
                Wallet.address == "test_wallet_address"
            ).first()
            if not wallet or wallet.reputation < 5.0:
                raise ValueError("Insufficient reputation for command execution")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        logger.log(
            f"Command executed: {command}, returncode: {result.returncode}",
            request_id=request_id
        )
        return {
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "request_id": request_id
        }
    except Exception as e:
        logger.log(f"Command error: {str(e)}", request_id=request_id)
        raise
