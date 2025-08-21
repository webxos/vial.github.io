from server.models.mcp_alchemist import McpAlchemist


def test_alchemist_initialization():
    alchemist = McpAlchemist()
    assert alchemist is not None
    assert alchemist.model == "claude-3-opus"


def test_alchemist_process():
    alchemist = McpAlchemist()
    result = alchemist.process({"text": "test input"})
    assert result["status"] == "processed"
    assert "output" in result
