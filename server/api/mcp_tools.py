from pydantic import BaseModel
from typing import Dict, Any
from server.logging import logger
import uuid


class MCPToolSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    required: list[str] = []
    returns: Dict[str, str]


class MCPTools:
    @staticmethod
    def vial_status_get() -> MCPToolSchema:
        """Returns the schema for vial.status.get tool."""
        return MCPToolSchema(
            name="vial.status.get",
            description="Retrieves the status of a vial by ID, including balance and active state.",
            parameters={
                "vial_id": {"type": "string", "description": "Unique identifier of the vial"}
            },
            required=["vial_id"],
            returns={"status": "object"}
        )

    @staticmethod
    def vial_config_generate() -> MCPToolSchema:
        """Returns the schema for vial.config.generate tool."""
        return MCPToolSchema(
            name="vial.config.generate",
            description="Generates configuration for a vial based on provided parameters.",
            parameters={
                "vial_id": {"type": "string", "description": "Unique identifier of the vial"},
                "config_type": {"type": "string", "description": "Type of configuration (e.g., quantum, deploy)"}
            },
            required=["vial_id", "config_type"],
            returns={"config": "object"}
        )

    @staticmethod
    def quantum_circuit_build() -> MCPToolSchema:
        """Returns the schema for quantum.circuit.build tool."""
        return MCPToolSchema(
            name="quantum.circuit.build",
            description="Builds a quantum circuit using Qiskit for specified parameters.",
            parameters={
                "qubits": {"type": "integer", "description": "Number of qubits"},
                "gates": {"type": "array", "description": "List of quantum gates"}
            },
            required=["qubits"],
            returns={"circuit": "string"}
        )

    @staticmethod
    def deploy_vercel() -> MCPToolSchema:
        """Returns the schema for deploy.vercel tool."""
        return MCPToolSchema(
            name="deploy.vercel",
            description="Deploys the project to Vercel with specified configuration.",
            parameters={
                "project_id": {"type": "string", "description": "Vercel project ID"},
                "env": {"type": "object", "description": "Environment variables"}
            },
            required=["project_id"],
            returns={"status": "string"}
        )

    @staticmethod
    def git_commit_push() -> MCPToolSchema:
        """Returns the schema for git.commit.push tool."""
        return MCPToolSchema(
            name="git.commit.push",
            description="Commits and pushes changes to a Git repository.",
            parameters={
                "repo_path": {"type": "string", "description": "Path to the Git repository"},
                "commit_message": {"type": "string", "description": "Commit message"}
            },
            required=["repo_path", "commit_message"],
            returns={"status": "string"}
        )

    @staticmethod
    async def execute_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the specified MCP tool with given parameters."""
        request_id = str(uuid.uuid4())
        try:
            from server.services.mcp_alchemist import Alchemist
            alchemist = Alchemist()
            if tool_name == "vial.status.get":
                result = await alchemist.get_vial_status(params["vial_id"])
                return result
            elif tool_name == "vial.config.generate":
                # Placeholder for config generation
                return {"config": {"vial_id": params["vial_id"], "type": params["config_type"]}}
            elif tool_name == "quantum.circuit.build":
                from qiskit import QuantumCircuit
                circuit = QuantumCircuit(params["qubits"])
                return {"circuit": str(circuit)}
            elif tool_name == "deploy.vercel":
                # Placeholder for Vercel deployment
                return {"status": "deployed"}
            elif tool_name == "git.commit.push":
                from git import Repo
                repo = Repo(params["repo_path"])
                repo.git.commit(m=params["commit_message"])
                repo.git.push()
                return {"status": "success"}
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.log(f"Tool execution error: {str(e)}", request_id=request_id)
            raise
