import logging
from typing import List
from pydantic import BaseModel
from pymongo import MongoClient
from server.security.wallet_crypto import WalletCrypto, WalletData

logger = logging.getLogger(__name__)

class DAOProposal(BaseModel):
    proposal_id: str
    description: str
    votes: Dict[str, bool]

class WebXOSWallet:
    def __init__(self):
        self.mongo = MongoClient("mongodb://mongo:27017").vial_mcp
        self.crypto = WalletCrypto()

    async def create_proposal(self, proposal: DAOProposal, signatures: List[str]) -> bool:
        """Create DAO proposal with M-of-N signatures."""
        try:
            wallet_data = WalletData(address="0x123", balance=100.0, signatures=signatures)
            if not self.crypto.verify_multi_sig(wallet_data, signatures, m=2):
                raise ValueError("Invalid signatures")
            self.mongo.proposals.insert_one(proposal.dict())
            return True
        except Exception as e:
            logger.error(f"Proposal creation failed: {str(e)}")
            raise

    async def export_wallet(self, wallet_id: str, password: str) -> str:
        """Export wallet as encrypted .md."""
        try:
            wallet = self.mongo.wallets.find_one({"_id": wallet_id})
            if not wallet:
                raise ValueError("Wallet not found")
            wallet_data = WalletData(address=wallet["address"], balance=wallet["balance"], signatures=[])
            return self.crypto.encrypt_wallet(wallet_data, password)
        except Exception as e:
            logger.error(f"Wallet export failed: {str(e)}")
            raise
