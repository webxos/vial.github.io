from swarm import Swarm, Agent
from server.services.mcp_alchemist import Alchemist
from server.api.mcp_tools import MCPTools
from pymongo import MongoClient
from server.logging import logger
import uuid
import os

class AgentTasks:
    def __init__(self):
        self.swarm_client = Swarm()
        self.mongo_client = MongoClient(os.getenv("MONGO_URL", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]

    async def execute_task(self, task: str, params: dict, alchemist: Alchemist = Depends(Alchemist)):
        request_id = str(uuid.uuid4())
        try:
            context = {"params": params}
            if task == "vial_status_get":
                agent = Agent(
                    name="WalletAgent",
                    instructions="Handle wallet tasks like balance checks.",
                    functions=[MCPTools.vial_status_get]
                )
                result = await self.swarm_client.run(
                    agent=agent,
                    messages=[{"role": "user", "content": f"Get vial status for {params.get('vial_id')}"}],
                    context_variables=context
                )
            elif task == "quantum_circuit_build":
                agent = Agent(
                    name="QuantumAgent",
                    instructions="Manage quantum circuit tasks using Qiskit.",
                    functions=[MCPTools.quantum_circuit_build]
                )
                result = await self.swarm_client.run(
                    agent=agent,
                    messages=[{"role": "user", "content": f"Build circuit with {params.get('qubits')} qubits"}],
                    context_variables=context
                )
            elif task == "git_commit_push":
                agent = Agent(
                    name="DeployAgent",
                    instructions="Handle deployment tasks to Git.",
                    functions=[MCPTools.git_commit_push]
                )
                result = await self.swarm_client.run(
                    agent=agent,
                    messages=[{"role": "user", "content": f"Commit to {params.get('repo_path')}"}],
                    context_variables=context
                )
            else:
                raise ValueError(f"Unknown task: {task}")

            self.db.tasks.insert_one({
                "task": task,
                "params": params,
                "result": result.messages[-1]["content"],
                "request_id": request_id,
                "timestamp": int(__import__('time').time())
            })
            logger.log(f"Task executed: {task}", request_id=request_id)
            return result.messages[-1]["content"]
        except Exception as e:
            logger.log(f"Task execution error: {str(e)}", request_id=request_id)
            self.db.errors.insert_one({
                "task": task,
                "error": str(e),
                "request_id": request_id,
                "timestamp": int(__import__('time').time())
            })
            raise
