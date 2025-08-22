from fastapi import FastAPI
from server.services.prompt_training import PromptTrainer
from server.services.agent_tasks import AgentTaskManager
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time


class TrainingScheduler:
    def __init__(self, app: FastAPI):
        self.app = app
        self.prompt_trainer = PromptTrainer(app)
        self.task_manager = AgentTaskManager(app)
        self.schedule = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics = {"tasks_executed": 0, "total_time": 0.0}

    async def schedule_training(self, task_id: str, interval: int, params: dict):
        while True:
            start_time = datetime.now()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, lambda: [
                self.task_manager.execute_task(task_id, params),
                self.prompt_trainer.train_prompt("Train {context}", params)
            ])
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["tasks_executed"] += 1
            self.metrics["total_time"] += elapsed
            await asyncio.sleep(max(0, interval - elapsed))

    def get_metrics(self):
        return self.metrics


def setup_training_scheduler(app: FastAPI):
    scheduler = TrainingScheduler(app)
    app.state.training_scheduler = scheduler

    @app.post("/agent/schedule")
    async def schedule_task(task_id: str, interval: int, params: dict):
        asyncio.create_task(scheduler.schedule_training(task_id, interval, params))
        return {"status": "scheduled", "task_id": task_id}

    @app.get("/agent/metrics")
    async def get_scheduler_metrics():
        return app.state.training_scheduler.get_metrics()
