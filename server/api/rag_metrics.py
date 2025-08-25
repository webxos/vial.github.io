from fastapi import APIRouter, Depends, Response
from prometheus_client import generate_latest, Counter, Gauge
from server.security.oauth2 import validate_token
from server.rag.rag_config import RAGConfig

router = APIRouter(prefix="/metrics/rag")
rag_config = RAGConfig()

RAG_QUERIES = Counter(
    'mcp_rag_queries_total',
    'Total RAG queries',
    ['query_type']
)

DOCUMENTS_INDEXED = Gauge(
    'mcp_rag_documents_indexed',
    'Number of documents indexed in RAG'
)

@router.get("")
async def rag_metrics(token: str = Depends(validate_token)):
    """Expose RAG metrics."""
    # Update gauge with current document count
    DOCUMENTS_INDEXED.set(await rag_config.db.count_documents({}))
    RAG_QUERIES.labels(query_type="semantic").inc()
    
    return Response(generate_latest(), media_type="text/plain")
