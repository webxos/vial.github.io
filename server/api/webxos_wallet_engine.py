import uuid
from server.models.webxos_wallet import WalletModel
from pydantic import BaseModel


class StakeRequest(BaseModel):
    user_id: str
    amount: float


def update_wallet_balance(user_id, amount):
    wallet = WalletModel(user_id=user_id, balance=amount,
                        network_id=str(uuid.uuid4()))
    wallet.balance += amount
    return wallet.balance


def stake_tokens(request: StakeRequest):
    wallet = WalletModel(user_id=request.user_id, balance=request.amount,
                        network_id=str(uuid.uuid4()))
    wallet.balance -= request.amount
    return {"status": "staked", "amount": request.amount}
