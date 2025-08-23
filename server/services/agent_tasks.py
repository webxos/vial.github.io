from typing import Dict, Any, List
import yaml
from pymongo import MongoClient
from server.services.mcp_alchemist import Alchemist
from server.logging_config import logger
import os
import uuid

class AgentTasks:
    def __init__(self):
        self.alchemist = Alchemist()
        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]
        self.tasks_config = self.load_tasks()
        self.global_config = self.load_global_config()

    def load_tasks(self) -> Dict[str, Any]:
        with open("server/config/tasks.yaml", "r") as f:
            return yaml.safe_load(f)

    def load_global_config(self) -> Dict[str, Any]:
        with open("server/config/mcp_alchemist.yaml", "r") as f:
            return yaml.safe_load(f)

    async def execute_task(self, task_name: str, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        try:
            for agent in self.tasks_config["agents"]:
                for task in agent["tasks"]:
                    if task["name"] == task_name:
                        params = self._replace_tags(task["params"], params)
                        result = await self.alchemist.delegate_task(task["method"], params)
                        self.db.pretrained_prompts.insert_one({
                            "task_name": task_name,
                            "params": params,
                            "result": result,
                            "timestamp": "2025-08-23T02:03:00Z"
                        })
                        logger.info(f"Task {task_name} executed for {agent['id']}", request_id=request_id)
                        return result
            raise ValueError(f"Task {task_name} not found")
        except Exception as e:
            logger.error(f"Task execution error: {str(e)}", request_id=request_id)
            raise

    async def execute_global_command(self, command_name: str, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        try:
            for command in self.global_config["global_commands"]:
                if command["command"] == command_name:
                    params = self._replace_tags(command["params"], params)
                    result = await self.alchemist.delegate_task(command["method"], params)
                    self.db.agent_logs.insert_one({
                        "command_name": command_name,
                        "params": params,
                        "result": result,
                        "timestamp": "2025-08-23T02:03:00Z"
                    })
                    logger.info(f"Global command {command_name} executed", request_id=request_id)
                    return result
            raise ValueError(f"Global command {command_name} not found")
        except Exception as e:
            logger.error(f"Global command error: {str(e)}", request_id=request_id)
            raise

    def _replace_tags(self, template: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        result = template.copy()
        for key, value in result.items():
            if isinstance(value, str):
                for param_key, param_value in params.items():
                    result[key] = value.replace(f"{{{{{param_key}}}}}", str(param_value))
        return result

    async def get_pretrained_prompt(self, task_name: str) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        try:
            prompt = self.db.pretrained_prompts.find_one({"task_name": task_name})
            if prompt:
                logger.info(f"Retrieved pretrained prompt for {task_name}", request_id=request_id)
                return prompt
            logger.info(f"No pretrained prompt for {task_name}", request_id=request_id)
            return {}
        except Exception as e:
            logger.error(f"Pretrained prompt error: {str(e)}", request_id=request_id)
            raise
