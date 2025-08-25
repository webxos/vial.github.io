from typing import Any
from fastapi import APIRouter, Depends
from server.api.auth_endpoint import verify_token
import importlib.util
import sys
from pathlib import Path

class LangChainService:
    def __init__(self):
        self.v1_spec = importlib.util.spec_from_file_location(
            "langchain_v1", str(Path("venv/lib/python3.11/site-packages/langchain_openai/__init__.py"))
        )
        self.v2_spec = importlib.util.spec_from_file_location(
            "langchain_v2", str(Path("venv/lib/python3.11/site-packages/langchain/__init__.py"))
        )
        self.v1_module = None
        self.v2_module = None

    def load_v1(self):
        if self.v1_module is None:
            self.v1_module = importlib.util.module_from_spec(self.v1_spec)
            sys.modules["langchain_v1"] = self.v1_module
            self.v1_spec.loader.exec_module(self.v1_module)
        return self.v1_module

    def load_v2(self):
        if self.v2_module is None:
            self.v2_module = importlib.util.module_from_spec(self.v2_spec)
            sys.modules["langchain_v2"] = self.v2_module
            self.v2_spec.loader.exec_module(self.v2_module)
        return self.v2_module

    async def process_openai_task(self, input_data: str) -> dict:
        langchain_openai = self.load_v1()
        # Example: Use langchain_openai for OpenAI-specific tasks
        return {"result": f"Processed with langchain-openai: {input_data}"}

    async def process_general_task(self, input_data: str) -> dict:
        langchain = self.load_v2()
        # Example: Use langchain for general LLM tasks
        return {"result": f"Processed with langchain: {input_data}"}

langchain_service = LangChainService()

router = APIRouter(prefix="/mcp/langchain", tags=["langchain"])

@router.post("/openai")
async def process_openai(input_data: str, token: dict = Depends(verify_token)) -> dict:
    return await langchain_service.process_openai_task(input_data)

@router.post("/general")
async def process_general(input_data: str, token: dict = Depends(verify_token)) -> dict:
    return await langchain_service.process_general_task(input_data)
