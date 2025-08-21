from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from server.models.webxos_wallet import WebXOSWallet
from server.services.logging import Logger
from server.services.security import verify_jwt
import os


router = APIRouter()
logger = Logger("upload")
wallet_manager = WebXOSWallet()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    network_id: str = Form(...),
    token: str = Depends(verify_jwt)
):
    try:
        os.makedirs("/uploads", exist_ok=True)
        file_path = f"/uploads/{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        await wallet_manager.update_wallet(
            network_id,
            {"action": "upload", "filename": file.filename}
        )
        await logger.info(f"File uploaded: {file_path} for network {network_id}")
        return {"filePath": file_path}
    except Exception as e:
        await logger.error(f"Upload error: {str(e)}")
        os.makedirs("db", exist_ok=True)
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.utcnow().isoformat()}]** Upload error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
