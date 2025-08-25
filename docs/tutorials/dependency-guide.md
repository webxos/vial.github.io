Dependency Management Guide for WebXOS 2025 Vial MCP SDK
This guide explains how to manage the LangChain dependency conflict in the WebXOS 2025 Vial MCP SDK, allowing the use of both langchain-openai==0.1.0 and langchain==0.2.0/langchain-community==0.2.0.
Dependency Conflict Overview

Issue: langchain-openai==0.1.0 requires langchain-core<0.2.0, while langchain==0.2.0 and langchain-community==0.2.0 require langchain-core>=0.2.0.
Solution: Use two separate dependency files and dynamically load the appropriate LangChain version in langchain_service.py.

Setup Instructions

Install Base Dependencies:
pip install -r requirements-mcp.txt


Install LangChain Dependencies:

For OpenAI tasks:pip install -r requirements-langchain-v1.txt


For general LLM tasks:pip install -r requirements-langchain-v2.txt




Environment Configuration:Update .env with OpenAI API key:
OPENAI_API_KEY=your-openai-api-key



Using LangChain Service

OpenAI Tasks: Use /mcp/langchain/openai endpoint.
General LLM Tasks: Use /mcp/langchain/general endpoint.
Example:curl -H "Authorization: Bearer <token>" -X POST http://localhost:8000/mcp/langchain/openai -d '{"input_data": "test"}'



Testing
Run tests to verify compatibility:
pytest server/tests/test_python_compat.py

CI Pipeline
The CI pipeline tests both dependency sets:

requirements-langchain-v1.txt
requirements-langchain-v2.txt

See .github/workflows/ci.yml for details.
