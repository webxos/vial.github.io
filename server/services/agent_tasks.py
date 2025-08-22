from fastapi import FastAPI
from server.services.vial_manager import VialManager
from server.logging import logger
import asyncio


def setup_agent_tasks(app: FastAPI):
    vial_manager = VialManager()

    @app.post("/agent/task")
    async def assign_task(vial_id: str, task: dict):
        if vial_id not in vial_manager.agents:
            logger.log(f"Agent not found: {vial_id}")
            return {"error": "Agent not found"}
        try:
            agent = vial_manager.agents[vial_id]["model"]
            task_type = task.get("type")
            task_data = task.get("data", {})
            result = await asyncio.to_thread(run_task, agent, task_type, task_data)
            logger.log(f"Task assigned to {vial_id}: {task_type}")
            return {"status": "completed", "result": result}
        except Exception as e:
            logger.log(f"Task error for {vial_id}: {str(e)}")
            return {"error": str(e)}


def run_task(agent, task_type: str, task_data: dict):
    if task_type == "predict":
        input_data = task_data.get("input")
        if input_data:
            import torch
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            with torch.no_grad():
                output = agent(input_tensor)
            return output.tolist()
    return {"error": "Invalid task type"}
