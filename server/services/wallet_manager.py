from typing import Dict, Any
from server.webxos_wallet import WebXOSWallet
from server.services.mcp_alchemist import Alchemist
from server.logging_config import logger
import uuid
import re

class WalletManager:
    def __init__(self):
        self.wallet = WebXOSWallet()
        self.alchemist = Alchemist()

    async def import_wallet(self, markdown: str, request_id: str) -> Dict[str, Any]:
        try:
            network_id = re.search(r"\*\*User ID\*\*: ([\w-]+)", markdown).group(1)
            balance = float(re.search(r"\*\*Balance\*\*: ([\d.]+)", markdown).group(1))
            vials = re.search(r"\*\*Vials\*\*: ([\w\s,]+)", markdown).group(1).split(", ")
            self.wallet.import_wallet(markdown)
            await self.alchemist.coordinate_agents({"network_id": network_id}, request_id)
            logger.info(f"Imported wallet for {network_id}", request_id=request_id)
            return {"network_id": network_id, "balance": balance, "vials": vials, "request_id": request_id}
        except Exception as e:
            logger.error(f"Wallet import error: {str(e)}", request_id=request_id)
            raise

    async def export_wallet(self, network_id: str, request_id: str) -> Dict[str, Any]:
        try:
            markdown = f"""
# Vial Wallet Export
**Timestamp**: 2025-08-23T01:54:00Z
**User ID**: {network_id}
**Wallet Address**: {network_id}
**Balance**: 764.0000 $WEBXOS
**Vials**: vial1, vial2, vial3, vial4
"""
            logger.info(f"Exported wallet for {network_id}", request_id=request_id)
            return {"markdown": markdown, "request_id": request_id}
        except Exception as e:
            logger.error(f"Wallet export error: {str(e)}", request_id=request_id)
            raise
