from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from server.services.vial_manager import VialManager
from server.services.quantum_sync import QuantumVisualSync
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
from datetime import datetime, timedelta
from git import Repo
from passlib.context import CryptContext


class ModelRouter:
    def __init__(self):
        self.provider = settings.MODEL_PROVIDER
        self.primary_model = settings.PRIMARY_MODEL
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=self.primary_model, api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "xai":
            from langchain_community.llms import Grok
            self.llm = Grok(api_key=settings.XAI_API_KEY)
        else:
            self.llm = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def run(self, prompt: str):
        return await self.llm.apredict(prompt)


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
    quantum_sync = QuantumVisualSync(vial_manager)
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
            name="quantum.sync.state",
            func=lambda x: quantum_sync.sync_quantum_state(x),
            description="Sync quantum state for a vial"
        )
    ]
    agent = initialize_agent(tools, model_router.llm, agent="zero-shot-react-description", verbose=True)

    async def verify_scopes(token: str, required_scopes: list):
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            token_scopes = payload.get("scopes", [])
            if not all(scope in token_scopes for scope in required_scopes):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return payload.get("sub")
        except JWTError as e:
            logger.log(f"JWT error: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token")

    @app.post("/alchemist/auth/token")
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
        try:
            user = db.query(WalletModel).filter(WalletModel.user_id == form_data.username).first()
            if not user or not pwd_context.verify(form_data.password, user.hashed_password):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            token_data = {
                "sub": form_data.username,
                "exp": datetime.utcnow() + timedelta(hours=1),
                "scopes": ["wallet:read", "wallet:sign", "git:push", "vercel:deploy"]
            }
            token = jwt.encode(token_data, settings.JWT_SECRET, algorithm="HS256")
            logger.log(f"Generated token for user: {form_data.username}")
            return {"access_token": token, "token_type": "bearer"}
        except Exception as e:
            logger.log(f"Token generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/alchemist/auth/me")
    async def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        user_id = await verify_scopes(token, ["wallet:read"])
        logger.log(f"User profile accessed: {user_id}")
        return {"user_id": user_id}

    @app.post("/alchemist/train")
    async def train_vials(prompt: str, vial_id: str = None, config: VisualConfig = None, db: Session = Depends(get_db), token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        await verify_scopes(token, ["vial:train"])
        try:
            if vial_id and vial_id not in vial_manager.agents:
                raise ValueError(f"Vial {vial_id} not found")
            input_data = torch.rand(100)
            with torch.no_grad():
                prediction = alchemist_model(input_data)
            result = await model_router.run(prompt)
            if config:
                config_id = str(uuid.uuid4())
                db.add(VisualConfig(id=config_id, name=f"config_{prompt[:50]}", components=config.components, connections=config.connections))
                db.commit()
                quantum_result = quantum_sync.create_quantum_circuit_from_visual(config.components)
                logger.log(f"Alchemist trained with config: {prompt}, quantum hash: {quantum_result['quantum_hash']}")
                return {"status": "trained", "result": result, "config_id": config_id, "quantum_hash": quantum_result["quantum_hash"]}
            logger.log(f"Alchemist trained: {prompt}")
            return {"status": "trained", "result": result, "prediction": prediction.tolist()}
        except Exception as e:
            logger.log(f"Alchemist training error: {str(e)}")
            return {"error": str(e)}

    @app.post("/alchemist/wallet/export")
    async def export_wallet(user_id: str, svg_style: str = "default", db: Session = Depends(get_db), token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        await verify_scopes(token, ["wallet:export"])
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
            logger.log(f"Exported wallet to {resource_path} for user: {user_id}")
            return {"status": "exported", "resource_path": resource_path}
        except Exception as e:
            logger.log(f"Wallet export error: {str(e)}")
            return {"error": str(e)}

    @app.post("/alchemist/copilot")
    async def copilot_generate(prompt: str, token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        await verify_scopes(token, ["copilot:generate"])
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
                logger.log(f"Copilot generated code for user")
                return {"status": "generated", "code": code, "commit": commit_result}
        except Exception as e:
            logger.log(f"Copilot error: {str(e)}")
            return {"error": str(e)}

    @app.post("/alchemist/troubleshoot")
    async def troubleshoot(error: str, token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        await verify_scopes(token, ["ops:troubleshoot"])
        try:
            options = [
                "1. Check database connection",
                "2. Restart vial agents",
                "3. Clear Redis cache",
                "4. View logs",
                "5. Revert to last backup"
            ]
            response = await model_router.run(f"Troubleshoot error: {error}. Suggest steps from: {options}")
            logger.log(f"Troubleshooting initiated for error: {error}")
            return {"status": "troubleshooting", "steps": response, "options": options}
        except Exception as e:
            logger.log(f"Troubleshoot error: {str(e)}")
            return {"error": str(e)}

    @app.post("/alchemist/git")
    async def run_git_command(command: str, token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        await verify_scopes(token, ["git:push"])
        try:
            result = repo.git.execute(["git"] + command.split())
            logger.log(f"Git command executed: {command}")
            return {"status": "executed", "result": result}
        except Exception as e:
            logger.log(f"Git command error: {str(e)}")
            return {"error": str(e)}

async def generate_config_from_prompt(prompt: str) -> dict:
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
        return {"components": components, "connections": []}
    except Exception as e:
        logger.log(f"Config generation error: {str(e)}")
        raise

async def deploy_to_vercel(config: dict, client: httpx.AsyncClient) -> dict:
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
        logger.log(f"Vercel deployment initiated: {response.json()['id']}")
        return {"status": "deployed", "deployment_id": response.json()["id"]}
    except Exception as e:
        logger.log(f"Vercel deployment error: {str(e)}")
        return {"error": str(e)}

async def commit_to_git(data: dict, repo: Repo, client: httpx.AsyncClient) -> dict:
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
        logger.log(f"Created PR for commit: {data.get('message')}")
        return {"status": "committed", "resource_path": output_path}
    except Exception as e:
        logger.log(f"Git commit error: {str(e)}")
        return {"error": str(e)}

def generate_svg_diagram(vial_id: str, style: str = "default") -> str:
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
        logger.log(f"Generated SVG at {resource_path}")
        return resource_path
