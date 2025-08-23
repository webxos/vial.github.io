from fastapi import FastAPI, Depends, HTTPException, WebSocket
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from langchain.agents import initialize_agent, Tool
from server.services.vial_manager import VialManager
from server.quantum.qiskit_engine import QiskitEngine
from server.models.visual_components import VisualConfig
from server.models.webxos_wallet import WalletModel
from server.services.database import get_db
from server.config import settings
from server.logging import logger
import torch
import torch.nn as nn
import httpx
import json
import uuid
import os
import re
from datetime import datetime, timedelta
from git import Repo
from passlib.context import CryptContext
from jose import jwt, JWTError
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis


tool_calls = Counter("mcp_tool_calls", "MCP tool calls", ["tool"])
tool_duration = Histogram("mcp_tool_duration_seconds", "MCP tool execution time", ["tool"])


class ModelRouter:
    def __init__(self):
        self.provider = settings.MODEL_PROVIDER
        self.primary_model = settings.PRIMARY_MODEL
        self.fallback_providers = ["xai", "openai"] if self.provider == "anthropic" else ["anthropic"]
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=self.primary_model, api_key=settings.ANTHROPIC_API_KEY, streaming=True)
        elif self.provider == "xai":
            from langchain_community.llms import Grok
            self.llm = Grok(api_key=settings.XAI_API_KEY, streaming=True)
        else:
            from langchain.llms import OpenAI
            self.llm = OpenAI(api_key=settings.OPENAI_API_KEY, streaming=True)

    def redact_pii(self, text: str) -> str:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        text = re.sub(email_pattern, '[REDACTED_EMAIL]', text)
        text = re.sub(phone_pattern, '[REDACTED_PHONE]', text)
        return text

    @retry(stop дру=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def run(self, prompt: str):
        prompt = self.redact_pii(prompt)
        async for chunk in self.llm.astream(prompt):
            yield self.redact_pii(chunk)
        for fallback in self.fallback_providers:
            try:
                if fallback == "xai":
                    from langchain_community.llms import Grok
                    llm = Grok(api_key=settings.XAI_API_KEY, streaming=True)
                elif fallback == "openai":
                    from langchain.llms import OpenAI
                    llm = OpenAI(api_key=settings.OPENAI_API_KEY, streaming=True)
                async for chunk in llm.astream(prompt):
                    yield self.redact_pii(chunk)
                break
            except Exception as e:
                logger.log(f"Fallback {fallback} failed: {str(e)}", request_id=str(uuid.uuid4()))


class AlchemistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.sigmoid(self.output(x))


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def setup_mcp_alchemist(app: FastAPI):
    vial_manager = VialManager()
    model_router = ModelRouter()
    alchemist_model = AlchemistModel()
    qiskit_engine = QiskitEngine()
    repo = Repo(os.getcwd())
    async_client = httpx.AsyncClient(timeout=10.0)
    tools = [
        Tool(
            name="vial.status.get",
            func=lambda x: vial_manager.get_vial_status(x),
            description="Get status of a vial by ID"
        ),
        Tool(
            name="vial.config.generate",
            func=lambda x: generate_config_from_prompt(x),
            description="Generate visual config from prompt"
        ),
        Tool(
            name="deploy.vercel",
            func=lambda x: deploy_to_vercel(json.loads(x), async_client),
            description="Deploy to Vercel with config"
        ),
        Tool(
            name="git.commit.push",
            func=lambda x: commit_to_git(json.loads(x), repo, async_client),
            description="Commit code to GitHub"
        ),
        Tool(
            name="quantum.circuit.build",
            func=lambda x: qiskit_engine.build_circuit_from_components(json.loads(x)),
            description="Build quantum circuit from components"
        )
    ]
    agent = initialize_agent(tools, model_router.llm, agent="zero-shot-react-description", verbose=True)

    async def init_rate_limiter():
        redis_client = redis.from_url(settings.REDIS_URL)
        await FastAPILimiter.init(redis_client)

    app.add_event_handler("startup", init_rate_limiter)

    async def log_audit(tool_name: str, params: dict, result: dict, user_id: str):
        audit_entry = {
            "tool": tool_name,
            "params": params,
            "result": result,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open("audit.log", "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
        logger.log(f"Audit logged for {tool_name}", request_id=str(uuid.uuid4()))

    async def verify_scopes(token: str, required_scopes: list, request_id: str):
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            token_scopes = payload.get("scopes", [])
            if not all(scope in token_scopes for scope in required_scopes):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return payload.get("sub")
        except JWTError as e:
            logger.log(f"JWT error: {str(e)}", request_id=request_id)
            raise HTTPException(status_code=401, detail="Invalid token")

    @app.post("/alchemist/auth/token")
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
        request_id = str(uuid.uuid4())
        try:
            user = db.query(WalletModel).filter(WalletModel.user_id == form_data.username).first()
            if not user or not pwd_context.verify(form_data.password, user.hashed_password):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            token_data = {
                "sub": form_data.username,
                "exp": datetime.utcnow() + timedelta(hours=1),
                "scopes": ["wallet:read", "wallet:export", "git:push", "vercel:deploy", "vial:train", "ops:troubleshoot", "quantum:circuit"]
            }
            token = jwt.encode(token_data, settings.JWT_SECRET, algorithm="HS256")
            await log_audit("auth.token", {"username": form_data.username}, {"status": "success"}, form_data.username)
            logger.log(f"Generated token for user: {form_data.username}", request_id=request_id)
            return {"access_token": token, "token_type": "bearer"}
        except Exception as e:
            logger.log(f"Token generation error: {str(e)}", request_id=request_id)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/alchemist/auth/me")
    async def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        request_id = str(uuid.uuid4())
        with tool_duration.labels(tool="auth.me").time():
            tool_calls.labels(tool="auth.me").inc()
            user_id = await verify_scopes(token, ["wallet:read"], request_id)
            await log_audit("auth.me", {}, {"user_id": user_id}, user_id)
            logger.log(f"User profile accessed: {user_id}", request_id=request_id)
            return {"user_id": user_id}

    @app.post("/alchemist/train", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
    async def train_vials(prompt: str, vial_id: str = None, config: VisualConfig = None, db: Session = Depends(get_db), token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        request_id = str(uuid.uuid4())
        with tool_duration.labels(tool="train").time():
            tool_calls.labels(tool="train").inc()
            user_id = await verify_scopes(token, ["vial:train"], request_id)
            try:
                if vial_id and vial_id not in vial_manager.agents:
                    raise ValueError(f"Vial {vial_id} not found")
                input_data = torch.rand(100)
                with torch.no_grad():
                    prediction = alchemist_model(input_data)
                result = ""
                async for chunk in model_router.run(prompt):
                    result += chunk
                if config:
                    config_id = str(uuid.uuid4())
                    db.add(VisualConfig(id=config_id, name=f"config_{prompt[:50]}", components=config.components, connections=config.connections))
                    db.commit()
                    quantum_result = qiskit_engine.build_circuit_from_components(config.components)
                    await log_audit("train", {"prompt": prompt, "vial_id": vial_id, "config_id": config_id}, quantum_result, user_id)
                    logger.log(f"Alchemist trained with config: {prompt}, quantum hash: {quantum_result['quantum_hash']}", request_id=request_id)
                    return {"status": "trained", "result": result, "config_id": config_id, "quantum_hash": quantum_result["quantum_hash"]}
                await log_audit("train", {"prompt": prompt, "vial_id": vial_id}, {"result": result}, user_id)
                logger.log(f"Alchemist trained: {prompt}", request_id=request_id)
                return {"status": "trained", "result": result, "prediction": prediction.tolist()}
            except Exception as e:
                logger.log(f"Alchemist training error: {str(e)}", request_id=request_id)
                return {"error": str(e)}

    @app.post("/alchemist/wallet/export", dependencies=[Depends(RateLimiter(times=3, seconds=60))])
    async def export_wallet(user_id: str, svg_style: str = "default", db: Session = Depends(get_db), token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        request_id = str(uuid.uuid4())
        with tool_duration.labels(tool="wallet.export").time():
            tool_calls.labels(tool="wallet.export").inc()
            user_id = await verify_scopes(token, ["wallet:export"], request_id)
            try:
                wallet = db.query(WalletModel).filter(WalletModel.user_id == user_id).first()
                if not wallet:
                    hashed_password = pwd_context.hash("default_password")
                    wallet = WalletModel(user_id=user_id, balance=72017.0, network_id=str(uuid.uuid4()), hashed_password=hashed_password)
                    db.add(wallet)
                    db.commit()
                resource_path = f"resources/wallets/{user_id}_{datetime.utcnow().isoformat()}.json"
                os.makedirs(os.path.dirname(resource_path), exist_ok=True)
                export_data = {
                    "network_id": wallet.network_id,
                    "session_start": datetime.utcnow().isoformat() + "Z",
                    "reputation": 1229811727985,
                    "wallet": {
                        "key": str(uuid.uuid4()),
                        "balance": wallet.balance,
                        "address": str(uuid.uuid4()),
                        "hash": "042e2b6c16cc0471417e0bca0161be72258214efcf46953a63c6343b187887ce"
                    },
                    "vials": [
                        {
                            "name": f"vial{i}",
                            "status": vial_manager.get_vial_status(f"vial{i}")["status"],
                            "language": "Python",
                            "balance": 18004.25,
                            "address": str(uuid.uuid4()),
                            "hash": "042e2b6c16cc0471417e0bca0161be72258214efcf46953a63c6343b187887ce",
                            "svg_diagram": generate_svg_diagram(f"vial{i}", svg_style)
                        } for i in range(1, 5)
                    ]
                }
                with open(resource_path, "w") as f:
                    json.dump(export_data, f)
                await log_audit("wallet.export", {"user_id": user_id, "svg_style": svg_style}, {"resource_path": resource_path}, user_id)
                logger.log(f"Exported wallet to {resource_path} for user: {user_id}", request_id=request_id)
                return {"status": "exported", "resource_path": resource_path}
            except Exception as e:
                logger.log(f"Wallet export error: {str(e)}", request_id=request_id)
                return {"error": str(e)}

    @app.post("/alchemist/copilot", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
    async def copilot_generate(prompt: str, token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        request_id = str(uuid.uuid4())
        with tool_duration.labels(tool="copilot.generate").time():
            tool_calls.labels(tool="copilot.generate").inc()
            user_id = await verify_scopes(token, ["copilot:generate"], request_id)
            try:
                async with async_client as client:
                    response = await client.post(
                        "https://api.github.com/copilot/generate",
                        headers={"Authorization": f"Bearer {settings.GITHUB_TOKEN}"},
                        json={"prompt": prompt, "language": "python"}
                    )
                    response.raise_for_status()
                    code = response.json().get("code", "")
                    commit_result = await commit_to_git({"code": code, "message": f"Copilot: {prompt[:50]}"}, repo, async_client)
                    await log_audit("copilot.generate", {"prompt": prompt}, commit_result, user_id)
                    logger.log(f"Copilot generated code for user", request_id=request_id)
                    return {"status": "generated", "code": code, "commit": commit_result}
            except Exception as e:
                logger.log(f"Copilot error: {str(e)}", request_id=request_id)
                return {"error": str(e)}

    @app.post("/alchemist/troubleshoot", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
    async def troubleshoot(error: str, token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        request_id = str(uuid.uuid4())
        with tool_duration.labels(tool="ops.troubleshoot").time():
            tool_calls.labels(tool="ops.troubleshoot").inc()
            user_id = await verify_scopes(token, ["ops:troubleshoot"], request_id)
            try:
                options = [
                    "1. Check database connection",
                    "2. Restart vial agents",
                    "3. Clear Redis cache",
                    "4. View logs",
                    "5. Revert to last backup"
                ]
                result = ""
                async for chunk in model_router.run(f"Troubleshoot error: {error}. Suggest steps from: {options}"):
                    result += chunk
                await log_audit("ops.troubleshoot", {"error": error}, {"steps": result, "options": options}, user_id)
                logger.log(f"Troubleshooting initiated for error: {error}", request_id=request_id)
                return {"status": "troubleshooting", "steps": result, "options": options}
            except Exception as e:
                logger.log(f"Troubleshoot error: {str(e)}", request_id=request_id)
                return {"error": str(e)}

    @app.post("/alchemist/git", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
    async def run_git_command(command: str, token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        request_id = str(uuid.uuid4())
        with tool_duration.labels(tool="git.commit.push").time():
            tool_calls.labels(tool="git.commit.push").inc()
            user_id = await verify_scopes(token, ["git:push"], request_id)
            try:
                result = repo.git.execute(["git"] + command.split())
                await log_audit("git.commit.push", {"command": command}, {"result": result}, user_id)
                logger.log(f"Git command executed: {command}", request_id=request_id)
                return {"status": "executed", "result": result}
            except Exception as e:
                logger.log(f"Git command error: {str(e)}", request_id=request_id)
                return {"error": str(e)}

    @app.get("/alchemist/metrics")
    async def get_metrics(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        request_id = str(uuid.uuid4())
        with tool_duration.labels(tool="metrics").time():
            tool_calls.labels(tool="metrics").inc()
            user_id = await verify_scopes(token, ["wallet:read"], request_id)
            metrics = {
                "tool_calls": {metric["labels"]["tool"]: metric["value"] for metric in tool_calls._metrics},
                "avg_duration": {metric["labels"]["tool"]: metric["sum"] / max(1, metric["count"]) for metric in tool_duration._metrics}
            }
            await log_audit("metrics", {}, metrics, user_id)
            logger.log(f"Metrics retrieved for user: {user_id}", request_id=request_id)
            return metrics

    @app.websocket("/alchemist/ws")
    async def websocket_endpoint(websocket: WebSocket):
        request_id = str(uuid.uuid4())
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                tool_name = data.get("tool")
                params = data.get("params", {})
                user_id = await verify_scopes(data.get("token", ""), [f"{tool_name.replace('.', ':')}"], request_id)
                if tool_name in [tool.name for tool in tools]:
                    tool_calls.labels(tool=tool_name).inc()
                    with tool_duration.labels(tool=tool_name).time():
                        result = await tools[[t.name for t in tools].index(tool_name)].func(params)
                        await log_audit(tool_name, params, result, user_id)
                        await websocket.send_json({"status": "success", "result": result, "request_id": request_id})
                else:
                    await websocket.send_json({"error": f"Tool {tool_name} not found", "request_id": request_id})
        except Exception as e:
            logger.log(f"WebSocket error: {str(e)}", request_id=request_id)
            await websocket.send_json({"error": str(e), "request_id": request_id})
        finally:
            await websocket.close()

async def generate_config_from_prompt(prompt: str) -> dict:
    request_id = str(uuid.uuid4())
    try:
        components = [
            {
                "id": f"comp{i}",
                "type": "api_endpoint",
                "title": f"API {i}",
                "position": {"x": i * 10, "y": 0, "z": 0},
                "config": {"prompt": prompt},
                "connections": [],
                "svg_style": "default"
            } for i in range(1, 5)
        ]
        result = {"components": components, "connections": []}
        await log_audit("vial.config.generate", {"prompt": prompt}, result, "system")
        return result
    except Exception as e:
        logger.log(f"Config generation error: {str(e)}", request_id=request_id)
        raise

async def deploy_to_vercel(config: dict, client: httpx.AsyncClient) -> dict:
    request_id = str(uuid.uuid4())
    try:
        headers = {"Authorization": f"Bearer {settings.VERCEL_API_TOKEN}"}
        payload = {
            "name": "vial-mcp",
            "files": config.get("files", []),
            "projectId": settings.VERCEL_PROJECT_ID,
            "target": "production"
        }
        response = await client.post(
            f"https://api.vercel.com/v9/projects/{settings.VERCEL_PROJECT_ID}/deployments",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = {"status": "deployed", "deployment_id": response.json()["id"]}
        await log_audit("deploy.vercel", config, result, "system")
        logger.log(f"Vercel deployment initiated: {response.json()['id']}", request_id=request_id)
        return result
    except Exception as e:
        logger.log(f"Vercel deployment error: {str(e)}", request_id=request_id)
        return {"error": str(e)}

async def commit_to_git(data: dict, repo: Repo, client: httpx.AsyncClient) -> dict:
    request_id = str(uuid.uuid4())
    try:
        output_path = f"resources/code/{uuid.uuid4()}.py"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(data.get("code", ""))
        repo.git.add(output_path)
        repo.index.commit(data.get("message", "MCP Alchemist commit"))
        branch = f"feature/{uuid.uuid4().hex[:8]}"
        repo.git.checkout("-b", branch)
        await client.post(
            f"https://api.github.com/repos/{settings.GITHUB_REPO}/pulls",
            headers={"Authorization": f"Bearer {settings.GITHUB_TOKEN}"},
            json={"title": data.get("message"), "head": branch, "base": "main"}
        )
        result = {"status": "committed", "resource_path": output_path}
        await log_audit("git.commit.push", data, result, "system")
        logger.log(f"Created PR for commit: {data.get('message')}", request_id=request_id)
        return result
    except Exception as e:
        logger.log(f"Git commit error: {str(e)}", request_id=request_id)
        return {"error": str(e)}

def generate_svg_diagram(vial_id: str, style: str = "default") -> str:
    request_id = str(uuid.uuid4())
    try:
        resource_path = f"resources/svg/{vial_id}_{uuid.uuid4()}.svg"
        os.makedirs(os.path.dirname(resource_path), exist_ok=True)
        fill_color = "#3498db" if style == "default" else "#e74c3c" if style == "alert" else "#2ecc71"
        svg_content = f"""<svg width="200" height="200">
            <rect x="10" y="10" width="180" height="180" fill="{fill_color}"/>
            <text x="50" y="100" fill="white">{vial_id}</text>
        </svg>"""
        with open(resource_path, "w") as f:
            f.write(svg_content)
        logger.log(f"Generated SVG at {resource_path}", request_id=request_id)
        return resource_path
    except Exception as e:
        logger.log(f"SVG generation error: {str(e)}", request_id=request_id)
        raise
