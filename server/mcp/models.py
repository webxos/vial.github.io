from pydantic import BaseModel
from typing import Dict, List, Optional


class VialStatusInput(BaseModel):
    vial_id: str


class VialStatusOutput(BaseModel):
    vial_id: str
    status: str


class ConfigGenerateInput(BaseModel):
    prompt: str


class ConfigGenerateOutput(BaseModel):
    components: List[Dict]
    connections: List[Dict]


class DeployVercelInput(BaseModel):
    files: List[Dict]
    project_id: Optional[str] = None
    target: str = "production"


class DeployVercelOutput(BaseModel):
    status: str
    deployment_id: str


class GitCommitInput(BaseModel):
    code: str
    message: str


class GitCommitOutput(BaseModel):
    status: str
    resource_path: str


class QuantumCircuitInput(BaseModel):
    components: List[Dict]


class QuantumCircuitOutput(BaseModel):
    qasm: str
    quantum_hash: str


class WalletExportInput(BaseModel):
    user_id: str
    svg_style: Optional[str] = "default"


class WalletExportOutput(BaseModel):
    status: str
    resource_path: str


class TroubleshootInput(BaseModel):
    error: str


class TroubleshootOutput(BaseModel):
    status: str
    steps: str
    options: List[str]
