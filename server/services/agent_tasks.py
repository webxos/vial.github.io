from fastapi import FastAPI
import torch
from server.services.vial_manager import VialManager
from server.services.advanced_logging import AdvancedLogger


def setup_agent_tasks(app: FastAPI):
    logger = AdvancedLogger()
    vial_manager = VialManager()

    async def execute_task(vial_id: str, task: str):
        if vial_id not in vial_manager.agents:
            logger.log("Agent task failed: Vial not found", extra={"vial_id": vial_id})
            return {"error": "Vial not found"}
        agent = vial_manager.agents[vial_id]["model"]
        dummy_input = torch.randn(1, 10)
        output = agent(dummy_input)
        logger.log("Task executed", extra={"vial_id": vial_id, "task": task, "output": output.item()})
        return {"status": "completed", "vial_id": vial_id, "task": task, "output": output.item()}

    app.state.execute_task = execute_task
    logger.log("Agent tasks initialized", extra={"vials": len(vial_manager.agents)})
