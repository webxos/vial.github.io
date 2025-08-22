from fastapi import APIRouter, File, UploadFile
from server.logging import logger

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.log(f"Uploaded file: {file.filename}")
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}
