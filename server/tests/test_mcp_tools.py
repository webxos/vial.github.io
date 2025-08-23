import pytest
from server.api.mcp_tools import MCPTools
from server.services.mcp_alchemist import Alchemist
from server.models.webxos_wallet import WalletModel
from server.services.database import get_db
from sqlalchemy.orm import Session
import uuid


@pytest.mark.asyncio
async def test_vial_status_get(db: Session):
    vial_id = str(uuid.uuid4())
    wallet = WalletModel(vial_id=vial_id, balance=100.0, active=True)
    db.add(wallet)
    db.commit()

    alchemist = Alchemist()
    result = await alchemist.get_vial_status(vial_id, db)
    assert result["vial_id"] == vial_id
    assert result["balance"] == 100.0
    assert result["active"] is True


@pytest.mark.asyncio
async def test_mcp_tool_execution():
    params = {"vial_id": str(uuid.uuid4())}
    with pytest.raises(ValueError):  # Simulate missing vial
        await MCPTools.execute_tool("vial.status.get", params)

    params = {"qubits": 2, "gates": ["h", "cx"]}
    result = await MCPTools.execute_tool("quantum.circuit.build", params)
    assert "circuit" in result
    assert isinstance(result["circuit"], str)


@pytest.mark.asyncio
async def test_git_commit_push(tmp_path):
    from git import Repo
    repo_path = tmp_path / "test_repo"
    repo = Repo.init(repo_path)
    (repo_path / "test.txt").write_text("test")
    repo.git.add(all=True)

    params = {"repo_path": str(repo_path), "commit_message": "Test commit"}
    result = await MCPTools.execute_tool("git.commit.push", params)
    assert result["status"] == "success"
