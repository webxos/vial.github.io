# MCP Alchemist: Orchestrating Agent Guide (WebXOS 2025)

This document explains how to turn `mcp_alchemist` into a production‑grade Model Context Protocol (MCP) agent that can converse with Cursor/Claude Desktop, GitHub Copilot Agents, and other MCP clients; orchestrate tools, PyTorch training jobs, and Qiskit quantum logic; and run on GitHub + Vercel (or Docker) with OAuth2 wallet controls and SQLAlchemy.

Use this as a checklist and living runbook for your repo at `vial.github.io`.

---

## 1) What makes a “full MCP agent”

An MCP agent MUST:
- Expose an MCP server implementing the MCP JSON-RPC protocol over stdio, WebSocket, or HTTP.
- Declare capabilities: tools, resources (files), prompts, model backends.
- Provide schema’d tool definitions and streaming result channels.
- Authenticate (optional) and authorize tool access.
- Be discoverable by MCP clients (Cursor, Claude Desktop) via a config entry.

Recommended server frameworks:
- Lightweight: `mcp` Python SDK (Anthropic OSS), or `ai-mcp`/`mcp-server` TypeScript.
- FastAPI bridge: keep FastAPI for HTTP/API, but run an MCP server process that mounts the same tools.

Outcome: MCP clients can list tools, call them with typed params, stream results, and persist sessions.

---

## 2) Repo layout baseline

Suggested top-level directories/files to add or verify:
- `server/mcp/`
  - `server/mcp/server.py` – Python MCP server (stdio/ws) exporting Alchemist tools
  - `server/mcp/tools.py` – Tool adapters wrapping VialManager, Git, Vercel, Quantum sync, etc.
  - `server/mcp/models.py` – Pydantic schemas for MCP tool I/O
  - `server/mcp/auth.py` – OAuth2/JWT to MCP session mapping
- `server/quantum/`
  - `qiskit_engine.py` – circuit build/run, transpile, simulate, provider adapter
- `server/agents/`
  - `alchemist.py` – existing FastAPI integration, refactor logic to reusable services
- `server/services/` – existing
- `server/models/` – existing SQLAlchemy models
- `scripts/mcp_dev.sh`, `scripts/mcp_run.sh` – dev/run scripts
- `MCP_Alchemist_Integration_Guide.md` – this guide
- `mcp.toml` – MCP server manifest (see §6)

Keep FastAPI as your web control plane; MCP server is the agent I/O plane.

---

## 3) Core dependencies (Python)

Add to `requirements.txt` (or split into extras):
- MCP server + protocol
  - `anthropic-mcp>=0.2.0` (or `mcp>=0.1.0`) – official Python SDK
  - `websockets>=12.0` – if exposing WS
- Auth / security
  - `python-jose[cryptography]` (already used)
  - `passlib[bcrypt]` for hashed passwords
- LLM backends
  - `openai`, `anthropic`, `mistralai`, `google-generativeai`, `groq`, `xai` (pick what you need)
- Git / DevOps
  - `GitPython` (already used)
  - `httpx` for async HTTP
- Data & validation
  - `pydantic>=2` (FastAPI already brings it)
- Quantum
  - `qiskit`, `qiskit-aer`, optional: `qiskit-ibm-runtime`
- Async
  - `uvloop` (optional Linux perf)

Run:
```bash
pip install anthropic-mcp websockets qiskit qiskit-aer httpx mistralai google-generativeai groq xai
```

---

## 4) Security and wallet model

- Replace hard-coded credentials with DB + hashed passwords + OAuth2 device/PKCE.
- Wallet export/import: store signed `.md` bundles with detached signatures (e.g., ed25519) and encrypted keys at rest.
- Use scoped JWTs: `sub`, `scopes` (e.g., `wallet:read`, `wallet:sign`, `git:push`, `vercel:deploy`), short TTL, automatic rotation.
- Add policy checks in every tool execution path; deny by default.

---

## 5) Alchemist as an MCP Tool Suite

Wrap current functions behind typed MCP tools. Examples (names must be stable):
- `vial.status.get` – params: `{vial_id:string}` → status
- `vial.config.generate` – params: `{prompt:string}` → visual config
- `deploy.vercel` – params: `{files:FileRef[], project_id?:string, target?:"production"|"preview"}` → deployment info
- `git.commit.push` – params: `{path:string, message:string}`
- `quantum.sync.state` – params: `{vial_id:string}`
- `quantum.circuit.build` – params: `{components:Component[]}` → returns circuit qasm, hashes
- `wallet.export` – params: `{user_id:string, svg_style?:string}`
- `copilot.generate` – params: `{prompt:string, language?:string}`
- `ops.troubleshoot` – params: `{error:string}`
- `system.logs.tail` – params: `{service:string, n?:number}`

Return values should be JSON-serializable, with references to larger artifacts via `resources` (files) per MCP spec.

---

## 6) MCP server manifest (`mcp.toml`)

Add to repo root so Cursor/Claude Desktop can discover the server.

```toml
[name]
id = "mcp-alchemist"
label = "MCP Alchemist (WebXOS)"

[server]
# Choose one
command = "python"
args = ["-m", "server.mcp.server"]
transport = "stdio"

# Or expose WS if you deploy
# transport = "websocket"
# url = "wss://vial.github.io/mcp/ws"

[env]
PYTHONPATH = "."
VIAL_ENV = "prod"

[capabilities]
resources = true
prompts = true
tools = true

[permissions]
# Example granular grants for client UIs
"git.commit.push" = "prompt"
"deploy.vercel" = "prompt"
"wallet.export" = "allow"
```

Commit this file, then add a Cursor/Claude Desktop configuration entry pointing to it.

---

## 7) FastAPI ↔ MCP: single source of truth

- Extract business logic from FastAPI routes into `server/services/*` functions.
- Import those services both in FastAPI routes and MCP tools to avoid drift.
- Log via a unified logger; emit structured events for audits.

---

## 8) Qiskit quantum logic

Create `server/quantum/qiskit_engine.py` with:
- Builders translating `VisualConfig.components` into Qiskit circuits.
- Utilities: transpile levels, backends (Aer simulator by default), run and return results.
- Hashing: SHA-256 of circuit QASM + metadata; embed in wallet export.
- Optional: provider integration via `QISKIT_IBM_TOKEN` for real hardware.

Checklist:
- [ ] Map each visual “gate” node type to Qiskit op
- [ ] Validation of qubit/wire counts
- [ ] Circuit to/from QASM/JSON
- [ ] Deterministic hashing
- [ ] Unit tests for translation layer

---

## 9) LLM backend matrix (lightweight + OSS options)

Implement a pluggable “ModelRouter” service with the same interface and swap providers via env.

Supported backends (low-cost/light):
- Anthropic: `claude-3-haiku-latest`
- Mistral: `mistral-small`, `codestral-latest`
- OpenAI compatible OSS/API: `groq` (Llama3/Whisper), `xAI` (Grok), `Together`, `Fireworks`
- Google: `gemini-1.5-flash`
- Local: `Ollama` (Llama3, Qwen2.5), `vLLM` if you outgrow Ollama

Env sketch:
```bash
MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
XAI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
PRIMARY_MODEL=claude-3-haiku-latest
CODE_MODEL=codestral-latest
```

Router responsibilities:
- Retry with backoff, fallbacks, cost/time metrics
- Streaming to MCP tool result channels
- Safety/PII redaction hooks

---

## 10) GitHub + Vercel constraints

- Vercel builds Next.js UI; MCP server should run separately (Docker, Fly.io, Railway, or Vercel Functions with WS if required).
- For GitHub-only operation, provide `scripts/mcp_dev.sh` to run stdio MCP locally; clients spawn it.
- Store sensitive tokens as GitHub Actions secrets and Vercel environment variables.
- Provide a `vercel.json` rewrite for `/mcp/ws` if you proxy WS.

---

## 11) Observability

- Structured JSON logs with request IDs; MDC contains `user_id`, `wallet_id`, `vial_id`.
- Metrics: `prometheus_client` or OTEL; counters for tool calls, durations, error types.
- Audit trail: append-only store of tool invocations and results (hash large payloads).

---

## 12) Testing strategy

- Unit tests for each tool adapter (`pytest`)
- Contract tests for MCP tool schemas
- Integration tests for Git/Vercel deploy (recorded with `vcrpy` or sandbox projects)
- Qiskit circuit translation tests (golden QASM)

---

## 13) Hardening the provided `mcp_alchemist` snippet

Key fixes and enhancements you should implement:
- Replace `OpenAI` LangChain dependency with your `ModelRouter` to avoid single-vendor coupling.
- Add missing imports (`os`, `OAuth2PasswordBearer`) and remove placeholder creds; enforce hashed passwords.
- Use SQLAlchemy ORM for `VisualConfig` inserts (avoid raw SQL strings in route).
- Make `agent.run` non-blocking; prefer async tools and streaming responses.
- All external calls (GitHub, Vercel) should use `httpx.AsyncClient` with timeouts and retries.
- Scope JWTs with granular permissions by tool, checked inside each tool.
- Convert SVG generator into a resource writer that returns a file ref path for MCP clients.
- Do not push directly to `origin` without branch policy; support PR creation via GitHub API.
- Add rate limiting and anti-abuse for expensive tools (deploy, quantum run).

---

## 14) Minimal `server/mcp/server.py` outline

```python
import asyncio
from mcp.server import Server
from server.mcp.tools import build_tool_list

async def main():
    server = Server(name="mcp-alchemist")
    for tool in build_tool_list():
        server.add_tool(tool)
    await server.run_stdio()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 15) How to tag your GitHub repo for live study

- Add a topic tag on GitHub: `mcp-server`, `model-context-protocol`, `webxos`, `alchemist`, `qiskit`, `fastapi`, `pytorch`, `sqlalchemy`.
- Create a short `README` badge section with:
  - “MCP manifest”: link to `mcp.toml`
  - “Run MCP locally”: `scripts/mcp_dev.sh`
- Open a public issue “Architecture review requested” and pin it; include environment example; we can reference it in tooling.

---

## 16) End-to-end setup checklist

- [ ] Add `server/mcp/*` with tool adapters and stdio server
- [ ] Create `mcp.toml` manifest at repo root
- [ ] Implement `ModelRouter` with Anthropic/Mistral/Gemini/Groq/xAI/Ollama
- [ ] Add `server/quantum/qiskit_engine.py` and unit tests
- [ ] Replace hard-coded creds with OAuth2 + scoped JWTs + policies
- [ ] Implement resources storage for large artifacts (SVG, QASM, logs)
- [ ] Wire FastAPI services to MCP tools via shared service layer
- [ ] Provide `scripts/mcp_dev.sh` and `scripts/mcp_run.sh`
- [ ] Add CI: lint, tests, `mcp validate` (when available), build Docker image
- [ ] Add `vercel.json` rewrites if proxying WS; otherwise deploy MCP on Docker
- [ ] Document Cursor/Claude Desktop client config pointing to `mcp.toml`

---

## 17) Cursor/Claude Desktop client configuration

- In Cursor/Claude Desktop, add a custom MCP server pointing to the repo directory with `mcp.toml`, or to the running stdio command.
- Grant permissions interactively for `git.commit.push` and `deploy.vercel`.

Example `scripts/mcp_dev.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
python -m server.mcp.server
```

---

## 18) Roadmap to a scalable Alchemist

- Multi-tenant projects and wallets
- Task graph planner (DAG) with retries and checkpointing
- Dataset and artifact registry (MinIO/S3)
- Fine-tuning pipelines (LoRA/QLoRA) via PyTorch Lightning
- Cost governance and quota management per user/org
- Declarative policy as code for tool access

---

## 19) Appendix: Environment variables

```
JWT_SECRET=...
DATABASE_URL=postgresql+psycopg://...
REDIS_URL=redis://...
VERCEL_API_TOKEN=...
VERCEL_PROJECT_ID=...
GITHUB_TOKEN=...
QISKIT_IBM_TOKEN=...
MODEL_PROVIDER=anthropic
PRIMARY_MODEL=claude-3-haiku-latest
```

---

By following this guide, `mcp_alchemist` becomes a standards-compliant MCP orchestrator that can talk to Cursor, Claude, Copilot, and other agents, while coordinating your FastAPI, PyTorch, SQLAlchemy, Git, Vercel, and Qiskit components with robust security and observability.