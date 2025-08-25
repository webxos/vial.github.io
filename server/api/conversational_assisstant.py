from fastapi import APIRouter, Depends
from server.config.mcp_github_config import MCPGitHubConfig

router = APIRouter(prefix="/api/assistant")

@router.get("/query")
async def query_assistant(query: str, wallet_id: str, config: MCPGitHubConfig = Depends()):
    headers = config.get_headers()
    if not config.validate_wallet(wallet_id):
        return {"response": "Invalid wallet"}
    url = f"{config.mcp_url}tools/query"
    payload = {"query": query, "wallet": wallet_id}
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return {"response": resp.json().get("summary", "No summary available")}
