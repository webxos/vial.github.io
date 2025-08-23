from autogen import AssistantAgent, UserProxyAgent
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid

class AutoGenAgent:
    def __init__(self):
        self.alchemist = Alchemist()
        self.user_proxy = UserProxyAgent(name="UserProxy", human_input_mode="NEVER")
        self.assistant = AssistantAgent(name="MCPAssistant", llm_config={"model": "nanogpt"})

    async def process_wallet_task(self, wallet_data: dict, task: str) -> dict:
        request_id = str(uuid.uuid4())
        try:
            network_id = wallet_data.get("network_id")
            self.user_proxy.initiate_chat(
                self.assistant,
                message=f"Process task '{task}' for wallet {network_id}"
            )
            result = self.assistant.last_message()["content"]
            logger.log(f"AutoGen task processed: {task}", request_id=request_id)
            return {"result": result, "request_id": request_id}
        except Exception as e:
            logger.log(f"AutoGen task error: {str(e)}", request_id=request_id)
            raise
