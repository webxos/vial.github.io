import uuid
from server.models.webxos_wallet import WalletModel
from pydantic import BaseModel


class MultiSigRequest(BaseModel):
    user_ids: list[str]
    amount: float


def update_wallet_balance(user_id: str, amount: float) -> float:
    wallet = WalletModel(user_id=user_id, balance=amount,
                        network_id=str(uuid.uuid4()))
    wallet.balance += amount
    return wallet.balance


def stake_tokens(user_id: str, amount: float) -> dict:
    wallet = WalletModel(user_id=user_id, balance=amount,
                        network_id=str(uuid.uuid4()))
    wallet.balance -= amount
    return {"status": "staked", "amount": amount}


def multi_sig_transaction(request: MultiSigRequest) -> dict:
    if len(request.user_ids) < 2:
        return {"error": "Multi-signature requires at least 2 users"}
    return {"status": "approved", "amount": request.amount, "users": request.user_ids}
