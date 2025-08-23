from pydantic import BaseModel
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
from git import Repo
import os
import uuid


class Tool(BaseModel):
    name: str
    description: str
    handler: callable


async def vial_status_get(params: dict) -> dict:
    request_id = str(uuid.uuid4())
    try:
        vial_id = params.get("vial_id")
        alchemist = Alchemist()
        status = await alchemist.get_vial_status(vial_id)
        logger.log(f"Retrieved status for vial: {vial_id}", request_id=request_id)
        return {"status": status}
    except Exception as e:
        logger.log(f"Error getting vial status: {str(e)}", request_id=request_id)
        return {"error": str(e)}


async def vial_config_generate(params: dict) -> dict:
    request_id = str(uuid.uuid4())
    try:
        prompt = params.get("prompt")
        alchemist = Alchemist()
        config = await alchemist.generate_config(prompt)
        logger.log(f"Generated config from prompt", request_id=request_id)
        return {"config": config}
    except Exception as e:
        logger.log(f"Config generation error: {str(e)}", request_id=request_id)
        return {"error": str(e)}


async def deploy_vercel(params: dict) -> dict:
    request_id = str(uuid.uuid4())
    try:
        config = params.get("config")
        repo = Repo(os.getcwd())
        repo.git.add(all=True)
        repo.git.commit(m="Deploy to Vercel")
        repo.git.push()
        logger.log("Deployed to Vercel", request_id=request_id)
        return {"status": "deployed"}
    except Exception as e:
        logger.log(f"Vercel deployment error: {str(e)}", request_id=request_id)
        return {"error": str(e)}


async def git_commit_push(params: dict) -> dict:
    request_id = str(uuid.uuid4())
    try:
        message = params.get("message", "Automated commit")
        repo = Repo(os.getcwd())
        repo.git.add(all=True)
        repo.git.commit(m=message)
        repo.git.push()
        logger.log(f"Committed and pushed: {message}", request_id=request_id)
        return {"status": "committed"}
    except Exception as e:
        logger.log(f"Git commit error: {str(e)}", request_id=request_id)
        return {"error": str(e)}


async def quantum_circuit_build(params: dict) -> dict:
    request_id = str(uuid.uuid4())
    try:
        components = params.get("components", [])
        alchemist = Alchemist()
        circuit = await alchemist.build_quantum_circuit(components)
        logger.log("Built quantum circuit", request_id=request_id)
        return {"circuit": circuit}
    except Exception as e:
        logger.log(f"Quantum circuit build error: {str(e)}", request_id=request_id)
        return {"error": str(e)}


def build_tool_list() -> list[Tool]:
    return [
        Tool(
            name="vial.status.get",
            description="Get status of a vial by ID",
            handler=vial_status_get
        ),
        Tool(
            name="vial.config.generate",
            description="Generate visual config from prompt",
            handler=vial_config_generate
        ),
        Tool(
            name="deploy.vercel",
            description="Deploy to Vercel with config",
            handler=deploy_vercel
        ),
        Tool(
            name="git.commit.push",
            description="Commit code to GitHub",
            handler=git_commit_push
        ),
        Tool(
            name="quantum.circuit.build",
            description="Build quantum circuit from components",
            handler=quantum_circuit_build
        )
    ]
