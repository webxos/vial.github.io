from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from fastapi.security import OAuth2AuthorizationCodeBearer
from server.utils.security_sanitizer import sanitize_input

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class TopologyVisualizeRequest(BaseModel):
    qubits: int
    circuit_data: dict

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_topology_visualization(qubits: int, circuit_data: dict) -> str:
    """Generate SVG visualization for quantum topology."""
    try:
        sanitized_circuit = sanitize_input(circuit_data)
        circuit = QuantumCircuit(qubits)
        # Placeholder: Build circuit from sanitized data
        svg_output = circuit_drawer(circuit, output="mpl")._repr_svg_()
        return svg_output
    except Exception as e:
        logger.error(f"Topology visualization failed: {str(e)}")
        raise

@router.post("/mcp/topology/visualize")
async def visualize_topology(request: TopologyVisualizeRequest, token: str = Security(oauth2_scheme)):
    """Generate quantum topology visualization."""
    try:
        svg = await generate_topology_visualization(request.qubits, request.circuit_data)
        logger.info(f"Topology visualization generated: {request.qubits} qubits")
        return {"status": "success", "svg": svg}
    except Exception as e:
        logger.error(f"Topology visualization request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
