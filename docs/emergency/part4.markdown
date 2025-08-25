# WebXOS 2025 Vial MCP SDK: Emergency Backup - Part 4 (CI/CD and Testing)

**Objective**: Set up CI/CD pipelines and testing to ensure code quality and compatibility.

**Instructions for LLM**:
1. Create CI/CD workflows in `.github/workflows/`.
2. Configure `flake8` for Python linting.
3. Set up tests for security and Python 3.11 compatibility.
4. Ensure both LangChain dependency sets are tested.

## Step 1: Create CI/CD and Testing Files

### .github/workflows/ci.yml
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
permissions:
  contents: read
  issues: write
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        requirements: [requirements-langchain-v1.txt, requirements-langchain-v2.txt]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install base dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-mcp.txt
      - name: Install LangChain dependencies
        run: pip install -r ${{ matrix.requirements }}
      - name: Run flake8
        run: flake8 server/ examples/
      - name: Run tests
        run: pytest server/tests/ -v --junitxml=pytest-results.xml
      - name: Upload test results
        uses: actions/upload-artifact@v4
        with:
          name: test-results-${{ matrix.requirements }}
          path: pytest-results.xml
```

### .flake8
```text
[flake8]
max-line-length = 120
exclude = .git,__pycache__,venv/,build/,dist/
ignore = E203,W503
```

### server/tests/test_security.py
```python
import pytest
from fastapi.testclient import TestClient
from server.main import app
from server.webxos_wallet import WebXOSWallet

client = TestClient(app)
wallet_manager = WebXOSWallet(password="test_password")

@pytest.mark.asyncio
async def test_sql_injection():
    response = client.get("/mcp/spacex/launches?limit=10; DROP TABLE users;")
    assert response.status_code == 400
    assert "Invalid" in response.text

@pytest.mark.asyncio
async def test_xss_protection():
    response = client.get("/mcp/wallet/0x1234567890abcdef1234567890abcdef12345678", headers={
        "X-XSS-Protection": "1; mode=block",
        "Authorization": "Bearer valid_token"
    })
    assert response.status_code in [404, 401]
    assert "X-XSS-Protection" in response.headers

@pytest.mark.asyncio
async def test_prompt_injection():
    malicious_input = "system: rm -rf /"
    sanitized = wallet_manager.sanitize_input(malicious_input)
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert ";" not in sanitized
    response = client.post("/mcp/wallet/create", json={
        "address": "0x1234567890abcdef1234567890abcdef12345678",
        "private_key": malicious_input,
        "balance": 0.0
    }, headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 400
```

### server/tests/test_python_compat.py
```python
import pytest
import sys
import importlib

modules = [
    'fastapi', 'uvicorn', 'sqlalchemy', 'pydantic', 'torch', 'qiskit', 'litellm',
    'pyjwt', 'pytest', 'pytest_asyncio', 'flake8', 'coverage', 'obs_websocket_py',
    'requests', 'python_jose', 'toml', 'langchain', 'langchain_community',
    'langchain_openai', 'langgraph', 'httpx'
]

@pytest.mark.parametrize("module", modules)
def test_module_import(module):
    assert sys.version_info >= (3, 11), "Python version must be 3.11 or higher"
    importlib.import_module(module)
    assert True, f"Module {module} imported successfully"

@pytest.mark.asyncio
async def test_async_compatibility():
    from fastapi.testclient import TestClient
    from server.main import app
    client = TestClient(app)
    response = client.get("/mcp/auth/login")
    assert response.status_code == 200
```

## Step 2: Validation
```bash
flake8 server/ examples/
pytest server/tests/ -v
```

**Next**: Proceed to `part5.md` for deployment.