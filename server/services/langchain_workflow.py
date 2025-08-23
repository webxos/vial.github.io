from langgraph.graph import StateGraph
from typing import Dict, Any
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid

class LangGraphWorkflow:
    def __init__(self):
        self.alchemist = Alchemist()
        self.graph = self.build_graph()

    def build_graph(self) -> StateGraph:
        graph = StateGraph()
        graph.add_node("train", self.train_node)
        graph.add_node("push", self.push_node)
        graph.add_edge("train", "push")
        graph.set_entry_point("train")
        return graph.compile()

    async def train_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        try:
            result = await self.alchemist.train_vial(state["params"], request_id)
            logger.info(f"LangGraph train node executed", request_id=request_id)
            return {"result": result, "request_id": request_id}
        except Exception as e:
            logger.error(f"LangGraph train error: {str(e)}", request_id=request_id)
            raise

    async def push_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        try:
            result = await self.alchemist.git_push(state["params"], request_id)
            logger.info(f"LangGraph push node executed", request_id=request_id)
            return {"result": result, "request_id": request_id}
        except Exception as e:
            logger.error(f"LangGraph push error: {str(e)}", request_id=request_id)
            raise

    async def execute_workflow(self, params: Dict[str, Any]) -> Dict:
        request_id = str(uuid.uuid4())
        try:
            result = await self.graph.run({"params": params})
            logger.info(f"LangGraph workflow executed", request_id=request_id)
            return {"result": result, "request_id": request_id}
        except Exception as e:
            logger.error(f"LangGraph workflow error: {str(e)}", request_id=request_id)
            with open("errorlog.md", "a") as f:
                f.write(f"- **[2025-08-23T01:00:00Z]** Workflow error: {str(e)}\n")
            raise
