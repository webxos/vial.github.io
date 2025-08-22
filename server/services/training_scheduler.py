from fastapi import FastAPI
import asyncio

class TrainingScheduler:
    def __init__(self, app: FastAPI):
        self.app = app
        self.scheduled_tasks = {}

    async def schedule_training(self, task_id: str, interval: int, params: dict):
        while True:
            await asyncio.sleep(interval)
            if task_id in self.scheduled_tasks:
                await self.app.state.agent_tasks.execute_task(task_id, params)

def setup_training_scheduler(app: FastAPI):
    scheduler = TrainingScheduler(app)
    app.state.training_scheduler = scheduler

    @app.post("/agent/schedule")
    async def schedule_training_endpoint(task_id: str, interval: int, params: dict):
        asyncio.create_task(scheduler.schedule_training(task_id, interval, params))
        return {"status": "scheduled", "task_id": task_id}
