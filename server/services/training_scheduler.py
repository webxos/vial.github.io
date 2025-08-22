import asyncio
from server.services.vial_manager import VialManager


async def schedule_training(vial_id):
    vial_manager = VialManager()
    await vial_manager.train_vial(vial_id)


async def run_scheduler():
    while True:
        await schedule_training("vial1")
        await asyncio.sleep(3600)
