from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from server.services.visualization_processor import VisualizationProcessor
from server.config.database import get_db
from server.models.dao_repository import DAORepository
from sqlalchemy.orm import Session
from fastapi.security import OAuth2AuthorizationCodeBearer
import logging

router = APIRouter()
logging.basicConfig(level=logging.INFO, filename="logs/visualization.log")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/mcp/auth/authorize",
    tokenUrl="/mcp/auth/token",
    scheme_name="OAuth2PKCE"
)

class VisualizationRequest(BaseModel):
    circuit_qasm: str = None
    topology_data: dict = None
    render_type: str = "svg"  # "svg" or "3d"
    wallet_id: str

@router.post("/render")
async def render_visualization(request: VisualizationRequest, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        if not request.circuit_qasm and not request.topology_data:
            raise HTTPException(status_code=400, detail="Circuit or topology data required")
        
        processor = VisualizationProcessor()
        visualization_data = await processor.process_visualization(
            circuit_qasm=request.circuit_qasm,
            topology_data=request.topology_data,
            render_type=request.render_type
        )
        
        # Award DAO points for visualization
        dao_repo = DAORepository(db)
        dao_repo.add_reputation(request.wallet_id, points=5)
        
        return {
            "visualization": visualization_data,
            "render_type": request.render_type,
            "reputation_points": dao_repo.get_reputation(request.wallet_id)
        }
    except ValueError as e:
        logging.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
