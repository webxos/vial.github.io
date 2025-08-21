from server.services.audit_log import AuditLog
import asyncio


class TaskQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.audit = AuditLog()

    async def add_task(self, task: dict):
        await self.queue.put(task)
        await self.audit.log_action(
            action="add_task",
            user_id="system",
            details={"task": task}
        )
        return {"status": "task_added"}

    async def process_tasks(self):
        while True:
            task = await self.queue.get()
            try:
                if task["type"] == "quantum":
                    from server.models.mcp_alchemist import MCPAlchemist
                    alchemist = MCPAlchemist()
                    result = await alchemist.predict_quantum_outcome(task["params"])
                    await self.audit.log_action(
                        action="process_quantum_task",
                        user_id="system",
                        details={"task": task, "result": result}
                    )
                self.queue.task_done()
            except Exception as e:
                await self.audit.log_action(
                    action="task_failed",
                    user_id="system",
                    details={"task": task, "error": str(e)}
                )
                self.queue.task_done()
