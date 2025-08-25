from fastapi import APIRouter, Depends
from server.security.oauth2 import validate_token
import os
import json

router = APIRouter(prefix="/cuda/test")

class TestRunnerConfig:
    def __init__(self):
        self.test_dir = os.getenv("MCP_TEST_DIR", "./build")
        self.output_dir = os.getenv("MCP_OUTPUT_DIR", "./test_results")
        self.config_file = os.getenv("MCP_TEST_CONFIG", "test_args.json")
        self.parallel_runs = int(os.getenv("MCP_PARALLEL", "4"))

    async def get_config(self, token: str = Depends(validate_token)):
        with open(self.config_file, "r") as f:
            return json.load(f)
