from pydantic import BaseModel
from server.services.git_trainer import GitTrainer
from server.quantum.quantum_sync import QuantumSync

class MCPAlchemist(BaseModel):
    git_trainer: GitTrainer
    quantum_sync: QuantumSync

    async def process_task(self, task: dict):
        if task["type"] == "git":
            result = await self.git_trainer.execute_task(
                task["action"], task["params"]
            )
        elif task["type"] == "quantum":
            result = await self.quantum_sync.run_circuit(
                task["circuit"], task.get("backend", "qasm_simulator")
            )
        else:
            raise ValueError("Invalid task type")
        return result

    class Config:
        arbitrary_types_allowed = True
