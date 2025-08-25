from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from server.rag.rag_config import RAGConfig
from server.security.oauth2 import validate_token
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/api/rag")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
rag_config = RAGConfig()

class QueryRequest(BaseModel):
    query: str
    limit: int = 5

@router.post("/query")
async def rag_query(request: QueryRequest, token: str = Depends(oauth2_scheme)):
    """Perform RAG-based semantic search for space science queries."""
    await validate_token(token)
    try:
        result = await rag_config.query_knowledge_base(request.query, request.limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@router.post("/add-documents")
async def add_documents(documents: List[Dict], token: str = Depends(oauth2_scheme)):
    """Add documents to RAG knowledge base."""
    await validate_token(token)
    try:
        from langchain.schema import Document
        docs = [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in documents]
        result = await rag_config.initialize_knowledge_base(docs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document addition failed: {str(e)}")
