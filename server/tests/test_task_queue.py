import pytest
from server.automation.task_queue import TaskQueue


@pytest.mark.asyncio
async def test_add_task():
    queue = TaskQueue()
    task = {"type": "quantum", "params": {"qubits": 2, "gates": []}}
    response = await queue.add_task(task)
    assert response["status"] == "task_added"
    assert queue.queue.qsize() == 1
