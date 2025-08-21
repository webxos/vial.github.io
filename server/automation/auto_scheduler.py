import asyncio
from server.services.notification import send_notification
from server.services.audit_log import AuditLog


class AutoScheduler:
    def __init__(self):
        self.audit = AuditLog()

    async def schedule_task(self, task: str, interval: int):
        while True:
            try:
                await self.execute_task(task)
                await self.audit.log_action(
                    action="scheduled_task",
                    user_id="system",
                    details={"task": task, "interval": interval}
                )
                await asyncio.sleep(interval)
            except Exception as e:
                await self.audit.log_action(
                    action="scheduled_task_failed",
                    user_id="system",
                    details={"task": task, "error": str(e)}
                )
                await asyncio.sleep(interval)

    async def execute_task(self, task: str):
        if task == "send_notification":
            await send_notification("Scheduled task executed", channel="in-app")
        else:
            raise ValueError("Unknown task")
