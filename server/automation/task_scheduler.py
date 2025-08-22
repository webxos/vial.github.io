from fastapi import FastAPI

async def setup_task_scheduler(app: FastAPI):
    print("Setting up task scheduler")


async def schedule_task(app: FastAPI):
    print("Task scheduled")
