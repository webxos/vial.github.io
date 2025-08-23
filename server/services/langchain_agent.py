from langchain.agents import AgentExecutor, Tool
from langchain.llms.base import LLM
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
from typing import List, Optional
import importlib
import pkgutil
import uuid
import json

class NanoGPTLLM(LLM):
    def _call(self, prompt: str, **kwargs) -> str:
        return f"Simulated NanoGPT response to: {prompt}"

    @property
    def _llm_type(self) -> str:
        return "nanogpt"

class LangChainAgent:
    def __init__(self):
        self.alchemist = Alchemist()

    def create_agent(self) -> AgentExecutor:
        llm = NanoGPTLLM()
        tools = self.load_tools()
        return AgentExecutor.from_agent_and_tools(agent=llm, tools=tools, verbose=True)

    def load_tools(self) -> List[Tool]:
        tools = []
        for _, name, _ in pkgutil.iter_modules(['prompts']):
            module = importlib.import_module(f'prompts.{name}')
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, '_is_mcp_prompt'):
                    tools.append(Tool(
                        name=attr_name,
                        func=lambda x: attr(x),
                        description=f"Prompt: {attr_name}"
                    ))
        return tools

    async def process_wallet_task(self, wallet_data: dict, task: str) -> dict:
        request_id = str(uuid.uuid4())
        try:
            network_id = wallet_data.get("network_id")
            agent = self.create_agent()
            prompt = f"Process task '{task}' for wallet {network_id}"
            result = await agent.arun(prompt)
            logger.log(f"LangChain task processed: {task}", request_id=request_id)
            return {"result": result, "request_id": request_id}
        except Exception as e:
            logger.log(f"LangChain task error: {str(e)}", request_id=request_id)
            raise
