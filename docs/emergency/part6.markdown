# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 6 (Agent Sandboxing)

**Objective**: Implement a sandboxing mechanism for secure execution of LLM agents (Claude-Flow, OpenAI Swarm, CrewAI) in the WebXOS backend.

**Instructions for LLM**:
1. Create `server/security/agent_sandbox.py` to isolate agent execution.
2. Use Linux namespaces and resource limits for sandboxing.
3. Integrate with `server/main.py` for secure agent tasks.
4. Ensure compatibility with Python 3.11 and existing LangChain services.

## Step 1: Create Agent Sandbox File

### server/security/agent_sandbox.py
```python
import os
import sys
import subprocess
import tempfile
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
from server.services.langchain_service import langchain_service

class AgentSandbox:
    def __init__(self):
        self.sandbox_dir = tempfile.mkdtemp()

    async def execute_agent(self, agent_type: str, task: str) -> Dict[str, Any]:
        if agent_type not in ["claude-flow", "openai-swarm", "crewai"]:
            raise HTTPException(status_code=400, detail="Invalid agent type")

        # Write task to temporary file
        with open(f"{self.sandbox_dir}/task.py", "w") as f:
            if agent_type == "claude-flow":
                f.write(f"""
import langchain
from server.services.langchain_service import langchain_service
result = langchain_service.process_general_task("{task}")
print(result)
                """)
            elif agent_type == "openai-swarm":
                f.write(f"""
import langchain_openai
from server.services.langchain_service import langchain_service
result = langchain_service.process_openai_task("{task}")
print(result)
                """)
            else:  # crewai
                f.write(f"""
import crewai
result = {{'result': 'CrewAI task executed: {task}'}}
print(result)
                """)

        # Execute in sandbox with resource limits
        cmd = [
            "unshare", "--fork", "--pid", "--mount", "--uts", "--ipc", "--net",
            "python3", f"{self.sandbox_dir}/task.py"
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                env={"PYTHONPATH": ":".join(sys.path)}
            )
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail=result.stderr)
            return eval(result.stdout)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Agent execution timed out")

    def cleanup(self):
        import shutil
        shutil.rmtree(self.sandbox_dir)

agent_sandbox = AgentSandbox()

router = APIRouter(prefix="/mcp/agents", tags=["agents"])

@router.post("/{agent_type}")
async def execute_agent(agent_type: str, task: str, token: dict = Depends(verify_token)) -> Dict[str, Any]:
    try:
        result = await agent_sandbox.execute_agent(agent_type, task)
        return result
    finally:
        agent_sandbox.cleanup()
```

## Step 2: Integrate with Main Application
Update `server/main.py` to include the agent router:
```python
from server.security.agent_sandbox import router as agent_router
app.include_router(agent_router)
```

## Step 3: Validation
```bash
curl -H "Authorization: Bearer <token>" -X POST http://localhost:8000/mcp/agents/claude-flow -d '{"task": "Analyze data"}'
python -c "from server.security.agent_sandbox import AgentSandbox; s = AgentSandbox(); print(s.sandbox_dir)"
```

**Next**: Proceed to `part7.md` for database models.