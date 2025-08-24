from typing import Dict, List
from server.services.visualization_processor import VisualizationProcessor
from server.models.dao_repository import DAORepository
from sqlalchemy.orm import Session
import logging
import asyncio

logging.basicConfig(level=logging.INFO, filename="logs/swarm_agent.log")

class SwarmAgent:
    def __init__(self, db: Session):
        self.db = db
        self.visualizer = VisualizationProcessor()
        self.dao_repo = DAORepository(db)

    async def process_swarm_task(self, task_type: str, data: Dict, wallet_id: str) -> Dict:
        """Process distributed swarm task (e.g., visualization, computation)."""
        try:
            if task_type == "visualization":
                result = await self.visualizer.process_visualization(
                    circuit_qasm=data.get("circuit_qasm"),
                    topology_data=data.get("topology_data"),
                    render_type=data.get("render_type", "svg")
                )
                self.dao_repo.add_reputation(wallet_id, points=10)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            return {
                "result": result,
                "reputation_points": self.dao_repo.get_reputation(wallet_id)
            }
        except Exception as e:
            logging.error(f"Swarm task error: {str(e)}")
            raise ValueError(f"Swarm task failed: {str(e)}")

    async def distribute_tasks(self, tasks: List[Dict], wallet_id: str) -> List[Dict]:
        """Distribute tasks across swarm nodes."""
        try:
            results = []
            for task in tasks:
                result = await self.process_swarm_task(task["type"], task["data"], wallet_id)
                results.append(result)
            return results
        except Exception as e:
            logging.error(f"Task distribution error: {str(e)}")
            raise ValueError(f"Task distribution failed: {str(e)}")
