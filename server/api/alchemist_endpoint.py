from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.services.alchemist_agent import AlchemistAgent
from server.config.database import get_db
from sqlalchemy.orm import Session
from fastapi.security import OAuth2AuthorizationCodeBearer
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO, filename="logs/alchemist.log")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/mcp/auth/authorize",
    tokenUrl="/mcp/auth/token",
    scheme_name="OAuth2PKCE"
)

class AlchemistRequest(BaseModel):
    query: str
    quantum_circuit: str
    max_results: int = 5

@router.post("/")
async def run_alchemist_workflow(request: AlchemistRequest, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        # Prompt Shield validation
        if any(word in request.query.lower() for word in ["malicious", "hack", "exploit"]):
            raise HTTPException(status_code=400, detail="Invalid query detected")
        
        agent = AlchemistAgent(db)
        result = await agent.run_workflow(request.query, request.quantum_circuit)
        return {"results": result["results"], "circuit": result["circuit"], "provider": result["provider"]}
    except Exception as e:
        logging.error(f"Alchemist error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
