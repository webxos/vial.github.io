from fastapi import FastAPI
from server.models.vial_manager import VialManager
from server.models.webxos_wallet import Wallet
from server.services.git_trainer import GitTrainer
import asyncio

class AgentTaskManager:
    def __init__(self, app: FastAPI):
        self.app = app
        self.vial_manager = VialManager()
        self.wallet = Wallet()
        self.git_trainer = GitTrainer()

    async def execute_task(self, task_id: str, params: dict):
        if task_id == "train_vial":
            await self.vial_manager.train_vial(params.get("vial_id"))
            await self.wallet.update_balance(params.get("user_id"), 10.0)
        elif task_id == "git_commit":
            await self.git_trainer.commit_changes(params.get("message"))
        return {"status": "completed", "task_id": task_id}

def setup_agent_tasks(app: FastAPI):
    manager = AgentTaskManager(app)
    app.state.agent_tasks = manager

    @app.post("/agent/task")
    async def run_task(task_id: str, params: dict):
        return await app.state.agent_tasks.execute_task(task_id, params)
