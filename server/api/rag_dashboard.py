from fastapi import APIRouter, Depends, Response
from prometheus_client import generate_latest
from server.rag.rag_config import RAGConfig
from server.security.oauth2 import validate_token
from pydantic import BaseModel

router = APIRouter(prefix="/api/rag/dashboard")

class DashboardRequest(BaseModel):
    query: str
    limit: int = 5

@router.post("")
async def rag_dashboard(request: DashboardRequest, token: str = Depends(validate_token)):
    """Provide RAG dashboard data with metrics."""
    await validate_token(token)
    result = await RAGConfig().query_knowledge_base(request.query, request.limit)
    return {
        "metrics": {
            "query_count": RAG_QUERIES._value.get(),  # From rag_metrics.py
            "documents_indexed": DOCUMENTS_INDEXED._value.get()
        },
        "results": result
    }
