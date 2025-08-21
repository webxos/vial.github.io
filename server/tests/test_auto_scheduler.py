import pytest
from server.automation.auto_scheduler import AutoScheduler
from server.services.notification import send_notification


@pytest.mark.asyncio
async def test_schedule_task():
    scheduler = AutoScheduler()
    task = "send_notification"
    interval = 1
    async with asyncio.timeout(2):
        await scheduler.schedule_task(task, interval)
    response = await send_notification("Test scheduled task", channel="in-app")
    assert response["status"] == "sent"
    assert response["message"] == "Test scheduled task"
