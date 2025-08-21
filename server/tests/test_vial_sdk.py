from server.sdk.vial_sdk import vial_sdk


def test_sdk_initialization():
    assert vial_sdk is not None
    assert vial_sdk.api_key is not None


def test_sdk_execute():
    result = vial_sdk.execute({"command": "test"})
    assert result["status"] == "success"
    assert "result" in result
