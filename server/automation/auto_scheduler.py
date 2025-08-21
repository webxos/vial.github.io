from server.services.git_trainer import git_trainer
from server.logging import logger
import schedule
import time


class AutoScheduler:
    def __init__(self):
        self.scheduler = schedule


    def schedule_task(self, task_name: str, interval: int):
        try:
            self.scheduler.every(interval).seconds.do(git_trainer.create_repo,
                                                     task_name)
            logger.info(f"Scheduled task: {task_name} every {interval} seconds")
        except Exception as e:
            logger.error(f"Failed to schedule task: {str(e)}")
            raise ValueError(f"Scheduling failed: {str(e)}")


    def run(self):
        while True:
            self.scheduler.run_pending()
            time.sleep(1)


auto_scheduler = AutoScheduler()
