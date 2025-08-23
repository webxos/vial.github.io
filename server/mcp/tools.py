from mcp import Tool, ToolParameter
from git import Repo
import os
from server.services.mcp_alchemist import Alchemist
from server.logging import logger
import uuid

class MCPTools:
    @staticmethod
    @Tool(
        name="vial_status_get",
        description="Retrieve status and balance of a vial by ID.",
        parameters=[
            ToolParameter(name="vial_id", type="string", description="Unique vial identifier")
        ]
    )
    async def vial_status_get(vial_id: str, alchemist: Alchemist = Depends(Alchemist)):
        request_id = str(uuid.uuid4())
        try:
            status = await alchemist.get_vial_status(vial_id)
            logger.log(f"Vial status retrieved for {vial_id}", request_id=request_id)
            return status
        except Exception as e:
            logger.log(f"Error retrieving vial status: {str(e)}", request_id=request_id)
            raise

    @staticmethod
    @Tool(
        name="quantum_circuit_build",
        description="Build a quantum circuit with specified qubits and gates.",
        parameters=[
            ToolParameter(name="qubits", type="integer", description="Number of qubits"),
            ToolParameter(name="gates", type="array", description="List of gate operations")
        ]
    )
    async def quantum_circuit_build(qubits: int, gates: list, alchemist: Alchemist = Depends(Alchemist)):
        request_id = str(uuid.uuid4())
        try:
            from qiskit import QuantumCircuit
            circuit = QuantumCircuit(qubits)
            for gate in gates:
                if gate == "h":
                    circuit.h(range(qubits))
                elif gate == "cx":
                    circuit.cx(0, 1)
            svg = circuit.draw(output="svg")
            logger.log("Quantum circuit built", request_id=request_id)
            return {"circuit": svg}
        except Exception as e:
            logger.log(f"Error building quantum circuit: {str(e)}", request_id=request_id)
            raise

    @staticmethod
    @Tool(
        name="git_commit_push",
        description="Commit and push changes to a Git repository.",
        parameters=[
            ToolParameter(name="repo_path", type="string", description="Path to Git repository"),
            ToolParameter(name="commit_message", type="string", description="Commit message")
        ]
    )
    async def git_commit_push(repo_path: str, commit_message: str):
        request_id = str(uuid.uuid4())
        try:
            repo = Repo(repo_path)
            repo.git.add(all=True)
            repo.index.commit(commit_message)
            repo.remotes.origin.push()
            logger.log(f"Git commit and push completed for {repo_path}", request_id=request_id)
            return {"status": "success"}
        except Exception as e:
            logger.log(f"Git commit/push error: {str(e)}", request_id=request_id)
            raise
