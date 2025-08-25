from fastapi import FastAPI
from server.api.auth_endpoint import router as auth_router
from server.webxos_wallet import WebXOSWallet
from fastapi import Depends, HTTPException
from server.api.auth_endpoint import verify_token
from pydantic import BaseModel

app = FastAPI(title="WebXOS 2025 Vial MCP SDK", version="1.2.0")
app.include_router(auth_router)

# Wallet instance
wallet_manager = WebXOSWallet(password="secure_wallet_password")

class WalletCreate(BaseModel):
    address: str
    private_key: str
    balance: float = 0.0

@app.post("/mcp/wallet/create", tags=["wallet"])
async def create_wallet(wallet_data: WalletCreate, token: dict = Depends(verify_token)):
    try:
        wallet = wallet_manager.create_wallet(
            address=wallet_manager.sanitize_input(wallet_data.address),
            private_key=wallet_manager.sanitize_input(wallet_data.private_key),
            balance=wallet_data.balance
        )
        return {"address": wallet.address, "balance": wallet.balance}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/mcp/wallet/{address}", tags=["wallet"])
async def get_wallet(address: str, token: dict = Depends(verify_token)):
    address = wallet_manager.sanitize_input(address)
    if address not in wallet_manager.wallets:
        raise HTTPException(status_code=404, detail="Wallet not found")
    wallet = wallet_manager.wallets[address]
    return {"address": wallet.address, "balance": wallet.balance}

@app.post("/mcp/wallet/export/{address}", tags=["wallet"])
async def export_wallet(address: str, filename: str, token: dict = Depends(verify_token)):
    address = wallet_manager.sanitize_input(address)
    filename = wallet_manager.sanitize_input(filename)
    try:
        wallet_manager.export_wallet(address, filename)
        return {"message": f"Wallet exported to {filename}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mcp/wallet/import", tags=["wallet"])
async def import_wallet(filename: str, token: dict = Depends(verify_token)):
    filename = wallet_manager.sanitize_input(filename)
    try:
        wallet = wallet_manager.import_wallet(filename)
        return {"address": wallet.address, "balance": wallet.balance}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
