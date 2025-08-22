from fastapi import FastAPI
from server.services.agent_tasks import AgentTaskManager
from server.services.training_scheduler import TrainingScheduler


class ErrorRecovery:
    def __init__(self, app: FastAPI):
        self.app = app
        self.task_manager = AgentTaskManager(app)
        self.scheduler = TrainingScheduler(app)

    async def recover_task(self, task_id: str, params: dict):
        try:
            await self.task_manager.execute_task(task_id, params)
        except Exception as e:
            if task_id == "train_vial":
                await self.scheduler.schedule_training(task_id, 300, params)
            return {"status": "recovered", "error": str(e)}

def setup_error_recovery(app: FastAPI):
    recovery = ErrorRecovery(app)
    app.state.error_recovery = recovery

    @app.post("/agent/recover")
    async def recover_task_endpoint(task_id: str, params: dict):
        return await app.state.error_recovery.recover_task(task_id, params)
