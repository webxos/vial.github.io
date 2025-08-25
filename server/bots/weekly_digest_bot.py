import requests
from server.config.mcp_github_config import MCPGitHubConfig
import schedule
import time

def generate_digest(config: MCPGitHubConfig, repo: str, wallet_id: str):
    headers = config.get_headers()
    url = f"{config.mcp_url}repos/{repo}/pulls?state=all&per_page=5"
    pulls = requests.get(url, headers=headers).json()
    digest = "# Weekly Digest\n\n"
    if not config.validate_wallet(wallet_id):
        digest += f"- Wallet {wallet_id} invalid\n"
    for pr in pulls:
        digest += f"- PR #{pr['number']}: {pr['title']} (Wallet: {wallet_id})\n"
    return digest

def post_to_slack(digest: str):
    url = os.getenv("SLACK_WEBHOOK")
    payload = {"text": digest}
    requests.post(url, json=payload)

def run_digest(repo: str, wallet_id: str):
    config = MCPGitHubConfig()
    digest = generate_digest(config, repo, wallet_id)
    post_to_slack(digest)

schedule.every().monday.at("09:00").do(run_digest, "webxos/webxos-vial-mcp", "default_wallet")
while True:
    schedule.run_pending()
    time.sleep(60)
