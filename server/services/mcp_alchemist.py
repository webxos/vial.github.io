from fastapi import HTTPException, Depends
from pydantic import BaseModel
from swarm import Swarm, Agent, Result
from server.logging import logger
from server.services.redis_cache import RedisCache
from server.models.webxos_wallet import WalletModel
from server.models.swarm_agent import SwarmAgentModel
from server.api.mcp_tools import MCPTools
from server.services.database import get_db
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential
import uuid
import json
import os


class SwarmAgentConfig(BaseModel):
    name: str
    instructions: str
    functions: list[str] = []


class Alchemist:
    def __init__(self):
        self.swarm_client = Swarm()
        self.cache = RedisCache()
        self.playwright_mcp_url = os.getenv("PLAYWRIGHT_MCP_URL", "http://localhost:8080/mcp/playwright")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_vial_status(self, vial_id: str, db: Session = Depends(get_db)):
        request_id = str(uuid.uuid4())
        try:
            wallet = db.query(WalletModel).filter(WalletModel.vial_id == vial_id).first()
            if not wallet:
                raise HTTPException(status_code=404, detail=f"Vial {vial_id} not found")
            status = {"vial_id": vial_id, "balance": wallet.balance, "active": wallet.active}
            await self.cache.set_cache(CacheEntry(key=f"vial:{vial_id}", value=status))
            logger.log(f"Vial status retrieved: {vial_id}", request_id=request_id)
            return status
        except Exception as e:
            logger.log(f"Vial status error: {str(e)}", request_id=request_id)
            raise HTTPException(status_code=500, detail=str(e))

    async def delegate_task(self, task: str, context: dict, db: Session = Depends(get_db)):
        request_id = str(uuid.uuid4())
        try:
            wallet_agent = Agent(
                name="WalletAgent",
                instructions="Handle wallet-related tasks like balance checks and configurations.",
                functions=[MCPTools.vial_status_get, MCPTools.vial_config_generate]
            )
            quantum_agent = Agent(
                name="QuantumAgent",
                instructions="Manage quantum circuit tasks using Qiskit.",
                functions=[MCPTools.quantum_circuit_build]
            )
            deploy_agent = Agent(
                name="DeployAgent",
                instructions="Handle deployment tasks to Vercel and Git operations.",
                functions=[MCPTools.deploy_vercel, MCPTools.git_commit_push]
            )

            def route_to_agent(task: str):
                if "wallet" in task.lower() or "vial" in task.lower():
                    return Result(value="Routing to WalletAgent", agent=wallet_agent)
                elif "quantum" in task.lower() or "circuit" in task.lower():
                    return Result(value="Routing to QuantumAgent", agent=quantum_agent)
                elif "deploy" in task.lower() or "git" in task.lower():
                    return Result(value="Routing to DeployAgent", agent=deploy_agent)
                return None

            router_agent = Agent(
                name="RouterAgent",
                instructions="Decompose tasks and route to specialized agents.",
                functions=[route_to_agent]
            )

            response = await self.swarm_client.run(
                agent=router_agent,
                messages=[{"role": "user", "content": task}],
                context_variables=context
            )
            result = response.messages[-1]["content"]
            await self.cache.set_cache(CacheEntry(key=f"task:{request_id}", value={"result": result}))
            logger.log(f"Task delegated: {task}", request_id=request_id)

            # Persist agent state
            db_agent = SwarmAgentModel(
                name=response.agent.name,
                instructions=response.agent.instructions,
                functions=[f.__name__ for f in response.agent.functions]
            )
            db.add(db_agent)
            db.commit()
            return result
        except Exception as e:
            logger.log(f"Task delegation error: {str(e)}", request_id=request_id)
            raise HTTPException(status_code=500, detail=str(e))
