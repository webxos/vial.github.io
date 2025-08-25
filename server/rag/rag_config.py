from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from typing import List

class RAGConfig:
    def __init__(self, mongo_uri: str = "mongodb://mongo:27017", db_name: str = "space_science_rag"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.db["knowledge_base"],
            embedding=self.embeddings,
            index_name="vector_index"
        )
    
    async def initialize_knowledge_base(self, documents: List[Document]):
        """Initialize RAG knowledge base with documents."""
        await self.vector_store.add_documents(documents)
        return {"status": "success", "documents_added": len(documents)}
    
    async def query_knowledge_base(self, query: str, k: int = 5):
        """Query RAG knowledge base for relevant documents."""
        results = await self.vector_store.similarity_search(query, k=k)
        return {
            "query": query,
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        }
