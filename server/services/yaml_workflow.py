import yaml
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid
import os

class YAMLWorkflow:
    def __init__(self):
        self.alchemist = Alchemist()

    async def execute_workflow(self, yaml_content: str, wallet_data: dict) -> dict:
        request_id = str(uuid.uuid4())
        try:
            workflow = yaml.safe_load(yaml_content)
            network_id = wallet_data.get("network_id")
            results = []
            for step in workflow.get("steps", []):
                task = step.get("task")
                params = step.get("params", {})
                result = await self.alchemist.delegate_task(task, {"params": params})
                results.append({"task": task, "result": result})
                logger.log(f"YAML workflow step executed: {task}", request_id=request_id)
            return {"results": results, "request_id": request_id}
        except Exception as e:
            logger.log(f"YAML workflow error: {str(e)}", request_id=request_id)
            raise

    async def save_workflow(self, yaml_content: str, workflow_id: str):
        request_id = str(uuid.uuid4())
        try:
            os.makedirs("workflows", exist_ok=True)
            with open(f"workflows/{workflow_id}.yaml", "w") as f:
                f.write(yaml_content)
            logger.log(f"YAML workflow saved: {workflow_id}", request_id=request_id)
            return {"status": "success", "request_id": request_id}
        except Exception as e:
            logger.log(f"YAML save error: {str(e)}", request_id=request_id)
            raise
