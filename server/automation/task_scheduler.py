from fastapi import FastAPI
import asyncio
from datetime import datetime

class TaskScheduler:
    def __init__(self, app: FastAPI):
        self.app = app
        self.tasks = {}

    async def schedule_task(self, task_id: str, interval: int, callback):
        while True:
            start_time = datetime.now()
            await callback()
            elapsed = (datetime.now() - start_time).total_seconds()
            await asyncio.sleep(max(0, interval - elapsed))

def setup_task_scheduler(app: FastAPI):
    scheduler = TaskScheduler(app)
    app.state.task_scheduler = scheduler

    @app.post("/automation/schedule")
    async def schedule_task_endpoint(task_id: str, interval: int):
        asyncio.create_task(scheduler.schedule_task(task_id, interval, lambda: None))
        return {"status": "scheduled", "task_id": task_id}
