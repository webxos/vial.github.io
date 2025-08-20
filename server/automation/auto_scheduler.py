import schedule
import time
from server.services.git_trainer import git_trainer
from server.api.quantum_endpoints import router as quantum_router
from fastapi import FastAPI
from server.config import get_settings

app = FastAPI()

class AutoScheduler:
    def __init__(self):
        self.settings = get_settings()

    def schedule_training(self, repo_url: str, data_path: str, interval_minutes: int = 60):
        schedule.every(interval_minutes).minutes.do(
            lambda: git_trainer.train_from_repo(repo_url, data_path)
        )
        return {"status": "training scheduled"}

    def schedule_sync(self, node_id: str, interval_minutes: int = 30):
        async def sync_task():
            await quantum_router.get("/sync")({"node_id": node_id, "id": 1})
        schedule.every(interval_minutes).minutes.do(sync_task)
        return {"status": "sync scheduled"}

    def run(self):
        while True:
            schedule.run_pending()
            time.sleep(60)

auto_scheduler = AutoScheduler()
