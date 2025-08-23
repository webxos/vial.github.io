from mcp import Tool, ToolParameter, ToolResult
from server.services.vial_manager import VialManager
from server.quantum.qiskit_engine import QiskitEngine
from server.services.mcp_alchemist import generate_config_from_prompt, deploy_to_vercel, commit_to_git
from server.logging import logger
import httpx
import json
import uuid
from datetime import datetime


async def log_audit(tool_name: str, params: dict, result: dict, user_id: str):
    audit_entry = {
        "tool": tool_name,
        "params": params,
        "result": result,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open("audit.log", "a") as f:
        f.write(json.dumps(audit_entry) + "\n")
    logger.log(f"Audit logged for {tool_name}", request_id=str(uuid.uuid4()))


def build_tool_list():
    vial_manager = VialManager()
    qiskit_engine = QiskitEngine()
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
            name="quantum.circuit.build",
            description="Build quantum circuit from components",
            parameters=[
                ToolParameter(name="components", type="array", description="List of visual components")
            ],
            handler=lambda params: qiskit_engine.build_circuit_from_components(params["components"])
        )
    ]
    logger.log(f"Built {len(tools)} MCP tools", request_id=str(uuid.uuid4()))
    return tools
