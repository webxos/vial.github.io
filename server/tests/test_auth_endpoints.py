```python
import pytest
from fastapi.testclient import TestClient
from server.main import app
from unittest.mock import patch

client = TestClient(app)

@pytest.mark.asyncio
async def test_auth_endpoint_success():
    """Test successful wallet authentication."""
    with patch("cryptography.hazmat.primitives.asymmetric.kyber.Kyber512.public_key_length", return_value=32), \
         patch("jwt.encode", return_value="mocked_token"):
        response = client.post("/mcp/auth", json={"wallet_id": "test_wallet", "public_key": "a" * 64})
        assert response.status_code == 200
        assert response.json() == {"access_token": "mocked_token"}

@pytest.mark.asyncio
async def test_auth_endpoint_invalid_key():
    """Test authentication with invalid public key."""
    with patch("cryptography.hazmat.primitives.asymmetric.kyber.Kyber512.public_key_length", return_value=32):
        response = client.post("/mcp/auth", json={"wallet_id": "test_wallet", "public_key": "invalid"})
        assert response.status_code == 400
        assert "Invalid public key" in response.json()["detail"]
```
