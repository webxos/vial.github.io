from fastapi import FastAPI
from server.models.webxos_wallet import WalletModel
from server.services.vial_manager import VialManager
from server.services.git_trainer import GitTrainer

class AgentTaskManager:
    def __init__(self, app: FastAPI):
        self.app = app
        self.vial_manager = VialManager()
        self.wallet = WalletModel()
        self.git_trainer = GitTrainer()

    async def execute_task(self, task_id: str, params: dict):
        if task_id == "train_vial":
            await self.vial_manager.train_vial(params["vial_id"])
        elif task_id == "update_wallet":
            self.wallet.user_id = params.get("user_id", "default_user")
            self.wallet.balance = params.get("balance", 0.0)
            self.wallet.network_id = params.get("network_id", "test_net")
        elif task_id == "git_commit":
            self.git_trainer.commit_changes(params.get("message", "Auto-commit"))

def setup_agent_tasks(app: FastAPI):
    manager = AgentTaskManager(app)
    app.state.agent_tasks = manager

    @app.post("/agent/task")
    async def execute_task_endpoint(task_id: str, params: dict):
        await app.state.agent_tasks.execute_task(task_id, params)
        return {"status": "completed", "task_id": task_id}
