from fastapi import FastAPI
from server.services.vial_manager import VialManager
from server.services.advanced_logging import AdvancedLogger
from apscheduler.schedulers.asyncio import AsyncIOScheduler


def setup_training_scheduler(app: FastAPI):
    logger = AdvancedLogger()
    vial_manager = VialManager()
    scheduler = AsyncIOScheduler()

    async def schedule_training():
        for vial_id in vial_manager.agents:
            result = await vial_manager.train_vial(vial_id)
            logger.log("Scheduled training completed", extra={"vial_id": vial_id, "result": result})

    scheduler.add_job(schedule_training, 'interval', minutes=60)
    scheduler.start()
    app.state.scheduler = scheduler
    logger.log("Training scheduler initialized", extra={"interval": "60 minutes"})
