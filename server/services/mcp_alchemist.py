from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from server.services.vial_manager import VialManager
from server.models.visual_components import VisualConfig
from server.models.webxos_wallet import WalletModel
from server.services.database import get_db
from server.config import settings
from server.logging import logger
import torch
import torch.nn as nn
import requests
import json
import uuid
from datetime import datetime, timedelta
from git import Repo


class AlchemistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.sigmoid(self.output(x))


def setup_mcp_alchemist(app: FastAPI):
    vial_manager = VialManager()
    llm = OpenAI(api_key=settings.OPENAI_API_KEY)
    alchemist_model = AlchemistModel()
    repo = Repo(os.getcwd())
    tools = [
        Tool(
            name="VialStatus",
            func=lambda x: vial_manager.get_vial_status(x),
            description="Get status of a vial by ID"
        ),
        Tool(
            name="GenerateConfig",
            func=lambda x: generate_config_from_prompt(x),
            description="Generate visual config from prompt"
        ),
        Tool(
            name="VercelDeploy",
            func=lambda x: deploy_to_vercel(json.loads(x)),
            description="Deploy to Vercel with config"
        ),
        Tool(
            name="GitCommit",
            func=lambda x: commit_to_git(json.loads(x)),
            description="Commit code to GitHub"
        )
    ]
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    @app.post("/alchemist/auth/token")
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        try:
            if form_data.username != "test" or form_data.password != "test":  # Replace with DB check
                raise HTTPException(status_code=401, detail="Invalid credentials")
            token_data = {"sub": form_data.username, "exp": datetime.utcnow() + timedelta(hours=1)}
            token = jwt.encode(token_data, settings.JWT_SECRET, algorithm="HS256")
            logger.log(f"Generated token for user: {form_data.username}")
            return {"access_token": token, "token_type": "bearer"}
        except Exception as e:
            logger.log(f"Token generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/alchemist/auth/me")
    async def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/alchemist/auth/token"))):
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")
            logger.log(f"User profile accessed: {user_id}")
            return {"user_id": user_id}
        except JWTError as e:
            logger.log(f"JWT error: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token")

    @app.post("/alchemist/train")
    async def train_vials(prompt: str, vial_id: str = None, config: VisualConfig = None, db: Session = Depends(get_db)):
        try:
            if vial_id and vial_id not in vial_manager.agents:
                raise ValueError(f"Vial {vial_id} not found")
            input_data = torch.rand(100)  # Placeholder for prompt encoding
            with torch.no_grad():
                prediction = alchemist_model(input_data)
            result = agent.run(prompt)
            if config:
                db.execute(
                    "INSERT INTO visual_configs (id, name, components, connections, created_at, updated_at) "
                    "VALUES (:id, :name, :components, :connections, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                    {
                        "id": str(uuid.uuid4()),
                        "name": f"config_{prompt[:50]}",
                        "components": json.dumps([c.dict() for c in config.components]),
                        "connections": json.dumps([c.dict() for c in config.connections])
                    }
                )
                db.commit()
                logger.log(f"Alchemist trained with config: {prompt}")
                return {"status": "trained", "result": result, "config_id": config.id}
            logger.log(f"Alchemist trained: {prompt}")
            return {"status": "trained", "result": result, "prediction": prediction.tolist()}
        except Exception as e:
            logger.log(f"Alchemist training error: {str(e)}")
            return {"error": str(e)}

    @app.post("/alchemist/wallet/export")
    async def export_wallet(user_id: str, db: Session = Depends(get_db)):
        try:
            wallet = db.query(WalletModel).filter(WalletModel.user_id == user_id).first()
            if not wallet:
                wallet = WalletModel(user_id=user_id, balance=72017.0, network_id=str(uuid.uuid4()))
                db.add(wallet)
                db.commit()
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
                        "status": vial_manager.get_vial_status(f"vial{i}"),
                        "language": "Python",
                        "balance": 18004.25,
                        "address": str(uuid.uuid4()),
                        "hash": "042e2b6c16cc0471417e0bca0161be72258214efcf46953a63c6343b187887ce",
                        "svg_diagram": generate_svg_diagram(f"vial{i}")
                    } for i in range(1, 5)
                ]
            }
            logger.log(f"Exported wallet with SVG for user: {user_id}")
            return export_data
        except Exception as e:
            logger.log(f"Wallet export error: {str(e)}")
            return {"error": str(e)}

    @app.post("/alchemist/copilot")
    async def copilot_generate(prompt: str, user=Depends(get_current_user)):
        try:
            response = requests.post(
                "https://api.github.com/copilot/generate",
                headers={"Authorization": f"Bearer {settings.GITHUB_TOKEN}"},
                json={"prompt": prompt, "language": "python"}
            )
            response.raise_for_status()
            code = response.json().get("code", "")
            commit_result = commit_to_git({"code": code, "message": f"Copilot: {prompt[:50]}"})
            logger.log(f"Copilot generated code for user: {user['user_id']}")
            return {"status": "generated", "code": code, "commit": commit_result}
        except Exception as e:
            logger.log(f"Copilot error: {str(e)}")
            return {"error": str(e)}

    @app.post("/alchemist/troubleshoot")
    async def troubleshoot(error: str, user=Depends(get_current_user)):
        try:
            options = [
                "1. Check database connection",
                "2. Restart vial agents",
                "3. Clear Redis cache",
                "4. View logs",
                "5. Revert to last backup"
            ]
            response = agent.run(f"Troubleshoot error: {error}. Suggest steps from: {options}")
            logger.log(f"Troubleshooting initiated for error: {error}")
            return {"status": "troubleshooting", "steps": response, "options": options}
        except Exception as e:
            logger.log(f"Troubleshoot error: {str(e)}")
            return {"error": str(e)}

    @app.post("/alchemist/git")
    async def run_git_command(command: str, user=Depends(get_current_user)):
        try:
            result = repo.git.execute(["git"] + command.split())
            logger.log(f"Git command executed: {command}")
            return {"status": "executed", "result": result}
        except Exception as e:
            logger.log(f"Git command error: {str(e)}")
            return {"error": str(e)}

def generate_config_from_prompt(prompt: str) -> dict:
    try:
        components = [
            {
                "id": f"comp{i}",
                "type": "api_endpoint",
                "title": f"API {i}",
                "position": {"x": i * 10, "y": 0, "z": 0},
                "config": {"prompt": prompt},
                "connections": []
            } for i in range(1, 5)
        ]
        return {"components": components, "connections": []}
    except Exception as e:
        logger.log(f"Config generation error: {str(e)}")
        raise

def deploy_to_vercel(config: dict) -> dict:
    try:
        headers = {"Authorization": f"Bearer {settings.VERCEL_API_TOKEN}"}
        payload = {
            "name": "vial-mcp",
            "files": config.get("files", []),
            "projectId": settings.VERCEL_PROJECT_ID,
            "target": "production"
        }
        response = requests.post(
            "https://api.vercel.com/v9/projects/{}/deployments".format(settings.VERCEL_PROJECT_ID),
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        logger.log(f"Vercel deployment initiated: {response.json()['id']}")
        return {"status": "deployed", "deployment_id": response.json()["id"]}
    except Exception as e:
        logger.log(f"Vercel deployment error: {str(e)}")
        return {"error": str(e)}

def commit_to_git(data: dict) -> dict:
    try:
        repo = Repo(os.getcwd())
        output_path = "generated_code.py"
        with open(output_path, "w") as f:
            f.write(data.get("code", ""))
        repo.git.add(output_path)
        repo.index.commit(data.get("message", "MCP Alchemist commit"))
        repo.remotes.origin.push()
        logger.log(f"Committed to Git: {data.get('message')}")
        return {"status": "committed"}
    except Exception as e:
        logger.log(f"Git commit error: {str(e)}")
        return {"error": str(e)}

def generate_svg_diagram(vial_id: str) -> str:
    try:
        return f"""<svg width="200" height="200">
            <rect x="10" y="10" width="180" height="180" fill="#3498db"/>
            <text x="50" y="100" fill="white">{vial_id}</text>
        </svg>"""
    except Exception as e:
        logger.log(f"SVG generation error: {str(e)}")
        raise
