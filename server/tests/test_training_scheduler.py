from fastapi import FastAPI
from server.services.prompt_training import PromptTrainer
from server.services.agent_tasks import AgentTaskManager
import asyncio
from datetime import datetime, timedelta

class TrainingScheduler:
    def __init__(self, app: FastAPI):
        self.app = app
        self.prompt_trainer = PromptTrainer(app)
        self.task_manager = AgentTaskManager(app)
        self.schedule = {}

    async def schedule_training(self, task_id: str, interval: int, params: dict):
        while True:
            await self.task_manager.execute_task(task_id, params)
            await self.prompt_trainer.train_prompt("Train {context}", params)
            await asyncio.sleep(interval)

def setup_training_scheduler(app: FastAPI):
    scheduler = TrainingScheduler(app)
    app.state.training_scheduler = scheduler

    @app.post("/agent/schedule")
    async def schedule_task(task_id: str, interval: int, params: dict):
        asyncio.create_task(scheduler.schedule_training(task_id, interval, params))
        return {"status": "scheduled", "task_id": task_id}
