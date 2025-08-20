from fastapi import HTTPException
from octokit import Octokit
from server.config import get_settings

settings = get_settings()

def validate_jsonrpc_request(request: dict):
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        raise HTTPException(status_code=400, detail="Invalid JSON-RPC version")
    if "method" not in request or "id" not in request:
        raise HTTPException(status_code=400, detail="Missing method or id")
    return True

def github_search(query: str, max_results: int = 3):
    try:
        client = Octokit(auth="token", token=os.getenv("GITHUB_TOKEN", settings.GITHUB_TOKEN))
        response = client.search.code(q=query, sort="indexed")
        return [{"file": item["name"], "content": item["text_matches"]} for item in response.json["items"][:max_results]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GitHub search error: {str(e)}")
