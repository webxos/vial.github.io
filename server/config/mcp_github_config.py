from fastapi import Depends
from server.security.oauth2 import validate_token
import os
import requests

class MCPGitHubConfig:
    def __init__(self, token: str = Depends(validate_token)):
        self.mcp_url = "https://api.githubcopilot.com/mcp/"
        self.token = token
        self.toolsets = ["repos", "issues", "pull_requests", "actions", "code_security", "wallets"]
        self.wallet_path = os.path.join(os.getcwd(), ".md_wallets")

    def get_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def is_enabled(self, toolset: str):
        return toolset in self.toolsets or os.getenv("GITHUB_TOOLSETS", "").split(",") == ["all"]

    def validate_wallet(self, wallet_id: str):
        wallet_file = os.path.join(self.wallet_path, f"{wallet_id}.md")
        if not os.path.exists(wallet_file):
            self.report_error(f"Invalid wallet: {wallet_id}")
            return False
        return True

    def report_error(self, message: str):
        headers = self.get_headers()
        data = {"error": message, "type": "wallet_validation"}
        requests.post(f"{self.mcp_url}alerts", headers=headers, json=data)
