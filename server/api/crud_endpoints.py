from fastapi import APIRouter, Depends, HTTPException
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

router = APIRouter(prefix="/v1/crud", tags=["crud"])


@router.post("/create_wallet")
async def create_wallet(wallet_data: dict, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        wallet_id = str(uuid.uuid4())
        await memory_manager.save_wallet(wallet_id, wallet_data, request_id)
        logger.info(f"Created wallet {wallet_id}", request_id=request_id)
        return {"wallet_id": wallet_id, "status": "created", "request_id": request_id}
    except Exception as e:
        logger.error(f"Wallet creation error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_wallet/{wallet_id}")
async def get_wallet(wallet_id: str, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        wallet = await memory_manager.get_wallet(wallet_id, request_id)
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found")
        logger.info(f"Retrieved wallet {wallet_id}", request_id=request_id)
        return {"wallet": wallet, "request_id": request_id}
    except Exception as e:
        logger.error(f"Wallet retrieval error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_wallet/{wallet_id}")
async def delete_wallet(wallet_id: str, memory_manager: MemoryManager = Depends()):
    request_id = str(uuid.uuid4())
    try:
        await memory_manager.delete_wallet(wallet_id, request_id)
        logger.info(f"Deleted wallet {wallet_id}", request_id=request_id)
        return {"status": "deleted", "request_id": request_id}
    except Exception as e:
        logger.error(f"Wallet deletion error: {str(e)}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))
