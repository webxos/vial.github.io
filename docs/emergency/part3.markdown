# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 3 (API Services)

**Objective**: Implement API services for NASA, SpaceX, GitHub, and LangChain, integrating with the main FastAPI application.

**Instructions for LLM**:
1. Create service files in `server/services/`.
2. Implement NASA, SpaceX, and GitHub API integrations.
3. Handle LangChain version conflict by isolating `langchain-openai` and `langchain`/`langchain-community`.
4. Integrate services into `server/main.py`.

## Step 1: Create Service Files

### server/services/nasa_service.py
```python
import httpx
import os
from typing import Dict
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token

class NASADataClient:
    def __init__(self):
        self.api_key = os.getenv("NASA_API_KEY")
        self.base_url = "https://api.nasa.gov"

    async def query_earthdata(self, bbox: list, temporal: str, collection: str) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/planetary/earth",
                params={"bbox": ",".join(map(str, bbox)), "temporal": temporal, "collection": collection, "api_key": self.api_key}
            )
            return response.json()

    async def fetch_gibs_imagery(self, layer: str, date: str) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml",
                params={"layer": layer, "time": date, "api_key": self.api_key}
            )
            return response.json()

nasa_client = NASADataClient()

router = APIRouter(prefix="/mcp/nasa", tags=["nasa"])

@router.get("/earthdata")
async def get_earthdata(bbox: str, temporal: str, collection: str, token: dict = Depends(verify_token)):
    bbox_list = [float(x) for x in bbox.split(",")]
    return await nasa_client.query_earthdata(bbox_list, temporal, collection)

@router.get("/gibs")
async def get_gibs_imagery(layer: str, date: str, token: dict = Depends(verify_token)):
    return await nasa_client.fetch_gibs_imagery(layer, date)
```

### server/services/spacex_service.py
```python
import httpx
from typing import Dict, List
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token

class SpaceXService:
    BASE_URL = "https://api.spacexdata.com/v4"

    async def get_launches(self, limit: int = 10) -> List[Dict]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/launches", params={"limit": limit})
            response.raise_for_status()
            return response.json()

    async def get_starlink_satellites(self) -> List[Dict]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.BASE_URL}/starlink")
            response.raise_for_status()
            return response.json()

spacex_service = SpaceXService()

router = APIRouter(prefix="/mcp/spacex", tags=["spacex"])

@router.get("/launches")
async def get_launches(limit: int = 10, token: dict = Depends(verify_token)) -> List[Dict]:
    return await spacex_service.get_launches(limit)

@router.get("/starlink")
async def get_starlink(token: dict = Depends(verify_token)) -> List[Dict]:
    return await spacex_service.get_starlink_satellites()
```

### server/services/github_service.py
```python
import httpx
import os
from typing import Dict
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token

class GitHubService:
    def __init__(self):
        self.base_url = os.getenv("GITHUB_HOST", "https://api.githubcopilot.com")
        self.token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

    async def get_repo(self, owner: str, repo: str) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/repos/{owner}/{repo}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            response.raise_for_status()
            return response.json()

github_service = GitHubService()

router = APIRouter(prefix="/mcp/github", tags=["github"])

@router.get("/repos/{owner}/{repo}")
async def get_repo(owner: str, repo: str, token: dict = Depends(verify_token)) -> Dict:
    return await github_service.get_repo(owner, repo)
```

### server/services/langchain_service.py
```python
from typing import Any
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token
import importlib.util
import sys
from pathlib import Path

class LangChainService:
    def __init__(self):
        self.v1_spec = importlib.util.spec_from_file_location(
            "langchain_v1", str(Path("venv/lib/python3.11/site-packages/langchain_openai/__init__.py"))
        )
        self.v2_spec = importlib.util.spec_from_file_location(
            "langchain_v2", str(Path("venv/lib/python3.11/site-packages/langchain/__init__.py"))
        )
        self.v1_module = None
        self.v2_module = None

    def load_v1(self):
        if self.v1_module is None:
            self.v1_module = importlib.util.module_from_spec(self.v1_spec)
            sys.modules["langchain_v1"] = self.v1_module
            self.v1_spec.loader.exec_module(self.v1_module)
        return self.v1_module

    def load_v2(self):
        if self.v2_module is None:
            self.v2_module = importlib.util.module_from_spec(self.v2_spec)
            sys.modules["langchain_v2"] = self.v2_module
            self.v2_spec.loader.exec_module(self.v2_module)
        return self.v2_module

    async def process_openai_task(self, input_data: str) -> dict:
        langchain_openai = self.load_v1()
        return {"result": f"Processed with langchain-openai: {input_data}"}

    async def process_general_task(self, input_data: str) -> dict:
        langchain = self.load_v2()
        return {"result": f"Processed with langchain: {input_data}"}

langchain_service = LangChainService()

router = APIRouter(prefix="/mcp/langchain", tags=["langchain"])

@router.post("/openai")
async def process_openai(input_data: str, token: dict = Depends(verify_token)) -> dict:
    return await langchain_service.process_openai_task(input_data)

@router.post("/general")
async def process_general(input_data: str, token: dict = Depends(verify_token)) -> dict:
    return await langchain_service.process_general_task(input_data)
```

## Step 2: Validation
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/mcp/nasa/earthdata?bbox=-180,-90,180,90&temporal=2023-01-01/2023-12-31&collection=MODIS
curl -H "Authorization: Bearer <token>" http://localhost:8000/mcp/spacex/launches?limit=5
curl -H "Authorization: Bearer <token>" http://localhost:8000/mcp/github/repos/webxos/webxos-vial-mcp
curl -H "Authorization: Bearer <token>" -X POST http://localhost:8000/mcp/langchain/openai -d '{"input_data": "test"}'
```

**Next**: Proceed to `part4.md` for CI/CD and testing.