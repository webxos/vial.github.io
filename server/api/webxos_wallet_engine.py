import uuid
from server.models.webxos_wallet import WalletModel
def update_wallet_balance(user_id, amount):
    wallet = WalletModel(user_id=user_id, balance=amount, network_id=str(uuid.uuid4()))
    wallet.balance += amount
    return wallet.balance
