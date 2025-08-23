from fastapi import APIRouter, UploadFile, Depends
from server.webxos_wallet import WebXOSWallet
from server.logging_config import logger
from fastapi.security import OAuth2PasswordBearer
import uuid

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.post("/upload/wallet")
async def upload_wallet(file: UploadFile, token: str = Depends(oauth2_scheme)):
    request_id = str(uuid.uuid4())
    try:
        if not file.filename.endswith(".md"):
            raise ValueError("File must be a .md wallet file")
        content = await file.read()
        markdown = content.decode("utf-8")
        wallet = WebXOSWallet()
        result = wallet.import_wallet(markdown)
        logger.info(f"Uploaded wallet for {result['network_id']}", request_id=request_id)
        return {"status": "success", "network_id": result["network_id"], "request_id": request_id}
    except Exception as e:
        logger.error(f"Wallet upload error: {str(e)}", request_id=request_id)
        return {"status": "error", "detail": str(e), "request_id": request_id}
