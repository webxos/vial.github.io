from server.logging import logger
import uuid


class WebXosWallet:
    def __init__(self):
        self.address = str(uuid.uuid4())


    def get_balance(self):
        try:
            balance = 100
            logger.info(f"Retrieved balance for {self.address}")
            return balance
        except Exception as e:
            logger.error(f"Failed to get balance: {str(e)}")
            raise ValueError(f"Balance retrieval failed: {str(e)}")


    def send_transaction(self, recipient: str, amount: int):
        try:
            if amount <= 0:
                raise ValueError("Invalid amount")
            result = {"status": "success", "tx_id": str(uuid.uuid4())}
            logger.info(f"Sent transaction to {recipient} for {amount}")
            return result
        except Exception as e:
            logger.error(f"Transaction failed: {str(e)}")
            raise ValueError(f"Transaction failed: {str(e)}")


webxos_wallet = WebXosWallet()
