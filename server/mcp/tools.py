from mcp import Tool, ToolParameter, ToolResult
from server.services.vial_manager import VialManager
from server.services.quantum_sync import QuantumVisualSync
from server.services.mcp_alchemist import generate_config_from_prompt, deploy_to_vercel, commit_to_git
from server.logging import logger
import httpx
import json


def build_tool_list():
    vial_manager = VialManager()
    quantum_sync = QuantumVisualSync(vial_manager)
    async_client = httpx.AsyncClient(timeout=10.0)
    repo = Repo(os.getcwd())

    tools = [
        Tool(
            name="vial.status.get",
            description="Get status of a vial by ID",
            parameters=[
                ToolParameter(name="vial_id", type="string", description="Vial identifier")
            ],
            handler=lambda params: vial_manager.get_vial_status(params["vial_id"])
        ),
        Tool(
            name="vial.config.generate",
            description="Generate visual config from prompt",
            parameters=[
                ToolParameter(name="prompt", type="string", description="Prompt for config generation")
            ],
            handler=lambda params: generate_config_from_prompt(params["prompt"])
        ),
        Tool(
            name="deploy.vercel",
            description="Deploy to Vercel with config",
            parameters=[
                ToolParameter(name="config", type="object", description="Deployment configuration")
            ],
            handler=lambda params: deploy_to_vercel(params["config"], async_client)
        ),
        Tool(
            name="git.commit.push",
            description="Commit code to GitHub",
            parameters=[
                ToolParameter(name="data", type="object", description="Code and commit message")
            ],
            handler=lambda params: commit_to_git(params["data"], repo, async_client)
        ),
        Tool(
            name="quantum.sync.state",
            description="Sync quantum state for a vial",
            parameters=[
                ToolParameter(name="vial_id", type="string", description="Vial identifier")
            ],
            handler=lambda params: quantum_sync.sync_quantum_state(params["vial_id"])
        )
    ]
    logger.log(f"Built {len(tools)} MCP tools")
    return tools
