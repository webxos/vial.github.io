from fastapi import APIRouter, Depends
from server.services.logging import Logger
from server.services.security import verify_jwt


router = APIRouter()
logger = Logger("help")


@router.get("/help")
async def help_endpoint(token: str = Depends(verify_jwt)):
    try:
        help_content = {
            "endpoints": {
                "/auth/token": "Obtain OAuth 2.0 token for authentication.",
                "/auth/generate-credentials": "Generate API credentials for LangChain integration.",
                "/quantum/execute": "Execute quantum circuit and return results.",
                "/export": "Export wallet data in markdown format.",
                "/import": "Import wallet data from markdown file.",
                "/void": "Reset network state and vials (admin only).",
                "/troubleshoot": "Retrieve system health and recent logs.",
                "/help": "Display this API documentation.",
                "/vials/train": "Train vials with provided content.",
                "/vials/reset": "Reset all vials."
            },
            "instructions": {
                "authentication": "Use /auth/token to get a JWT token, then include 'Bearer <token>' in Authorization header.",
                "wallet": "Export/import wallets to manage $WEBXOS balances and transactions.",
                "quantum": "Use /quantum/execute for Qiskit-based simulations.",
                "vials": "Train vials with /vials/train to update models and earn $WEBXOS."
            }
        }
        await logger.info(f"Help requested by user {token.get('user_id')}")
        return {"status": "ok", "help": help_content}
    except Exception as e:
        await logger.error(f"Help endpoint error: {str(e)}")
        os.makedirs("db", exist_ok=True)
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.utcnow().isoformat()}]** Help endpoint error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))
