from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from typing import Dict, List
import os

class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore = None
        self.qa_chain = None

    async def initialize_vectorstore(self, urls: List[str]):
        try:
            loader = WebBaseLoader(urls)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(splits, self.embeddings)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize vectorstore: {str(e)}")

    async def query(self, query: str) -> Dict:
        if not self.qa_chain:
            await self.initialize_vectorstore([
                "https://api.nasa.gov/",
                "https://api.github.com/repos/webxos/webxos-vial-mcp"
            ])
        try:
            result = self.qa_chain({"query": query})
            return {"answer": result["result"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

rag_service = RAGService()

router = APIRouter(prefix="/mcp/rag", tags=["rag"])

@router.get("/query")
async def query_rag(query: str, token: dict = Depends(verify_token)) -> Dict:
    return await rag_service.query(query)
