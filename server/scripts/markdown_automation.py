import requests
from server.config.mcp_github_config import MCPGitHubConfig
import os

def fetch_issues(config: MCPGitHubConfig, repo: str, label: str):
    headers = config.get_headers()
    url = f"{config.mcp_url}repos/{repo}/issues?labels={label}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def format_issue(issue, wallet_id: str):
    config = MCPGitHubConfig()
    if not config.validate_wallet(wallet_id):
        return ""
    return f"## {issue['title']}\n\n{issue['body']}\n\n*Issue #{issue['number']} - Wallet: {wallet_id}*"

def automate_markdown(repo: str, label: str, wallet_id: str, output_dir: str):
    config = MCPGitHubConfig()
    issues = fetch_issues(config, repo, label)
    for issue in issues:
        with open(os.path.join(output_dir, f"issue_{issue['number']}_{wallet_id}.md"), "w") as f:
            f.write(format_issue(issue, wallet_id))
    os.system(f"git add {output_dir} && git commit -m 'Automated Markdown with wallet {wallet_id}' && git push")
