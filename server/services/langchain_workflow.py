from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from server.logging_config import logger
import uuid

class LangChainWorkflow:
    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["task_name", "params"],
            template="Execute task {task_name} with parameters: {params}"
        )
        self.llm_chain = LLMChain(prompt=self.prompt, llm=None)  # Placeholder for LLM

    async def execute_workflow(self, task_name: str, params: dict, request_id: str = str(uuid.uuid4())) -> dict:
        try:
            result = await self.llm_chain.run({"task_name": task_name, "params": params})
            logger.info(f"Workflow executed for task {task_name} with result: {result}", request_id=request_id)
            return {"status": "success", "result": result, "request_id": request_id}
        except Exception as e:
            logger.error(f"Workflow execution error for task {task_name}: {str(e)}", request_id=request_id)
            raise

    async def validate_workflow(self, task_name: str, params: dict, request_id: str = str(uuid.uuid4())) -> dict:
        try:
            valid = bool(params.get("vial_id") and task_name in ["train_model", "create_agent"])
            logger.info(f"Workflow validation for task {task_name}: {'valid' if valid else 'invalid'}", request_id=request_id)
            return {"status": "valid" if valid else "invalid", "request_id": request_id}
        except Exception as e:
            logger.error(f"Workflow validation error for task {task_name}: {str(e)}", request_id=request_id)
            raise

    async def retry_workflow(self, task_name: str, params: dict, request_id: str = str(uuid.uuid4())) -> dict:
        try:
            for attempt in range(3):
                result = await self.execute_workflow(task_name, params, request_id)
                if result["status"] == "success":
                    logger.info(f"Workflow retry succeeded for task {task_name} on attempt {attempt + 1}", request_id=request_id)
                    return result
            logger.error(f"Workflow retry failed for task {task_name} after 3 attempts", request_id=request_id)
            return {"status": "failed", "request_id": request_id}
        except Exception as e:
            logger.error(f"Workflow retry error for task {task_name}: {str(e)}", request_id=request_id)
            raise
