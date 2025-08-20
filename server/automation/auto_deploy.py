import asyncio
import schedule
from ..sdk.vial_sdk import vial_sdk
from ..logging import logger
from ..config import config

class AutoDeploy:
    def __init__(self):
        schedule.every(30).minutes.do(self.check_and_deploy)
        self.image_tag = "vial_mcp_sdk:latest"

    async def check_and_deploy(self):
        logger.info("Checking for deployment updates")
        containers = vial_sdk.docker_client.containers.list()
        if not containers or any(c.status != "running" for c in containers):
            logger.info("Deploying new container")
            result = vial_sdk.deploy_container(self.image_tag)
            logger.info(f"Deployment result: {result}")
        else:
            logger.info("System already running, no deployment needed")

    async def run(self):
        while True:
            schedule.run_pending()
            await asyncio.sleep(1)

auto_deploy = AutoDeploy()
