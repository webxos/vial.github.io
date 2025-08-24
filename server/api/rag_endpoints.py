from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel
from httpx import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
from pymongo import MongoClient
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

class RAGRequest(BaseModel):
    query: str
    max_results: int = 5

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def perform_vector_search(query: str, max_results: int) -> list:
    """Perform vector search using MongoDB Atlas."""
    try:
        sanitized_query = sanitize_input(query)
        client = MongoClient(settings.MONGO_URI)
        collection = client["vial_mcp"]["rag_vectors"]
        results = collection.aggregate([
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
        return list(results)
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise

@router.post("/mcp/rag")
async def rag_query(request: RAGRequest, token: str = Security(oauth2_scheme)):
    """Perform RAG query with vector search and LLM generation."""
    try:
        # Perform vector search
        search_results = await perform_vector_search(request.query, request.max_results)
        
        # Generate LLM response (placeholder)
        async with AsyncClient() as client:
            response = await client.post(
                f"{settings.XAI_API_KEY}/generate",  # Placeholder endpoint
                json={"prompt": f"Context: {search_results}\nQuery: {request.query}"}
            )
            response.raise_for_status()
            result = response.json()
        
        logger.info(f"RAG query processed: {request.query}")
        return {"status": "success", "results": search_results, "llm_response": result}
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
