from octokit import Octokit
import json
from ..services.database import get_db

octokit = Octokit(auth="your-github-token-here")  # Replace with dynamic token logic

def validate_token(token: str):
    with get_db() as db:
        cursor = db.execute("SELECT expires FROM credentials WHERE token = ?", (token,))
        result = cursor.fetchone()
        if result and result[0] > "2025-08-20T14:09:00Z":
            return True
    return False

def push_to_github(repo: str, message: str):
    try:
        octokit.repos.createCommitComment({
            "owner": "vial",
            "repo": repo,
            "commit_sha": "main",
            "body": message
        })
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}
