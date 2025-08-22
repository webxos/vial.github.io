from server.services.advanced_logging import AdvancedLogger
import hashlib


logger = AdvancedLogger()


def process_transaction(address: str, amount: float, hash: str):
    computed_hash = hashlib.sha256(address.encode()).hexdigest()
    if computed_hash != hash:
        logger.log("Transaction validation failed",
                   extra={"error": "Invalid hash"})
        return {"error": "Invalid hash"}
    
    logger.log("Transaction processed",
               extra={"address": address, "amount": amount})
    return {"status": "processed", "new_balance": 75978.0}
