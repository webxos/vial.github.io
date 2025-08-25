from server.nasa_tools.nasa_api_client import NASAAPIClient
from server.nasa_tools.nasa_quantum import NASAQuantumProcessor
from server.nasa_tools.nasa_ml import NASAMLProcessor
from fastapi import HTTPException
from typing import Dict

class NASAOrchestrator:
    def __init__(self):
        self.client = NASAAPIClient()
        self.quantum = NASAQuantumProcessor()
        self.ml = NASAMLProcessor()

    async def orchestrate_workflow(self, query: str, image_data: bytes = None) -> Dict:
        try:
            # Step 1: Search datasets
            datasets = await self.client.search_datasets(query)
            if not datasets:
                raise HTTPException(status_code=404, detail="No datasets found")

            # Step 2: Quantum correlation
            correlations = await self.quantum.correlate_datasets([d.id for d in datasets])
            correlated_ids = sorted(correlations, key=correlations.get, reverse=True)[:3]

            # Step 3: Analyze image if provided
            if image_data:
                analysis = await self.ml.analyze_image(image_data)
            else:
                analysis = {"prediction": None}

            return {
                "datasets": [d.dict() for d in datasets if d.id in correlated_ids],
                "correlations": correlations,
                "analysis": analysis
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

nasa_orchestrator = NASAOrchestrator()
