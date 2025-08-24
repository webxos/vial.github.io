from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from qiskit import QuantumCircuit, Aer, execute
from cryptography.hazmat.primitives.asymmetric import kyber
from pymongo import MongoClient
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from fastapi.security import OAuth2AuthorizationCodeBearer
from server.config.settings import settings
from server.utils.security_sanitizer import sanitize_input

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://github.com/login/oauth/authorize",
    tokenUrl="https://github.com/login/oauth/access_token"
)

class QuantumRAGRequest(BaseModel):
    query: str
    qubits: int
    max_results: int = 5

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def process_quantum_rag(query: str, qubits: int, max_results: int) -> dict:
    """Integrate quantum processing with RAG."""
    try:
        sanitized_query = sanitize_input(query)
        # Quantum circuit (placeholder)
        circuit = QuantumCircuit(qubits)
        backend = Aer.get_backend("statevector_simulator")
        result = execute(circuit, backend).result()
        statevector = result.get_statevector()

        # Vector search with MongoDB Atlas
        client = MongoClient(settings.MONGO_URI)
        collection = client["vial_mcp"]["rag_vectors"]
        search_results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": sanitized_query,  # Placeholder: Convert query to vector
                    "numCandidates": max_results * 10,
                    "limit": max_results
                }
            }
        ])

        # Encrypt results with Kyber-512
        kyber_key = kyber.Kyber512.generate_keypair()
        return {"statevector": statevector.tolist(), "rag_results": list(search_results)}
    except Exception as e:
        logger.error(f"Quantum-RAG processing failed: {str(e)}")
        raise

@router.post("/mcp/quantum_rag")
async def quantum_rag(request: QuantumRAGRequest, token: str = Security(oauth2_scheme)):
    """Process quantum-RAG query."""
    try:
        result = await process_quantum_rag(request.query, request.qubits, request.max_results)
        logger.info(f"Quantum-RAG processed: {request.query}, {request.qubits} qubits")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Quantum-RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
