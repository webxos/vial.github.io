from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from server.services.advanced_logging import AdvancedLogger
from server.services.vial_manager import VialManager


def setup_task_scheduler(app: FastAPI):
    logger = AdvancedLogger()
    vial_manager = VialManager()
    scheduler = AsyncIOScheduler()

    async def run_automated_task():
        for vial_id in vial_manager.agents:
            result = await vial_manager.train_vial(vial_id)
            logger.log("Automated task executed", extra={"vial_id": vial_id, "result": result})

    scheduler.add_job(run_automated_task, 'interval', hours=1)
    scheduler.start()
    app.state.task_scheduler = scheduler
    logger.log("Task scheduler initialized", extra={"interval": "1 hour"})
