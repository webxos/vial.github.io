from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, List
import requests
import tenacity
from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import os
from server.services.cuda_quantum_rag import CUDAQuantumRAG

class NASADataset(BaseModel):
    id: str
    title: str
    metadata: Dict

class NASAAPIClient:
    def __init__(self):
        self.base_url = "https://cmr.earthdata.nasa.gov/search/"
        self.api_key = os.getenv("NASA_API_KEY")
        self.rag = CUDAQuantumRAG()

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
    async def search_datasets(self, keyword: str, page_size: int = 10) -> List[NASADataset]:
        try:
            params = {"q": keyword, "page_size": page_size}
            if self.api_key:
                params["api_key"] = self.api_key
            response = requests.get(f"{self.base_url}collections", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            datasets = [NASADataset(id=item["id"], title=item["title"], metadata=item) for item in data["items"]]
            return datasets
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"NASA API search failed: {str(e)}")

    async def quantum_enhanced_search(self, query: str) -> List[Dict]:
        datasets = await self.search_datasets(query)
        embeddings = [self.rag.embeddings.embed_documents([d.metadata.get("summary", "")])[0] for d in datasets]
        query_embedding = self.rag.embeddings.embed_documents([query])[0]
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        job = execute(circuit, Aer.get_backend("statevector_simulator"))
        statevector = job.result().get_statevector()
        similarities = [np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)) for emb in embeddings]
        return [{"id": d.id, "title": d.title, "similarity": sim} for d, sim in zip(datasets, similarities)]

nasa_client = NASAAPIClient()
