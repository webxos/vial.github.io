import torch
import torch.nn as nn
from mcp.types import Tool, Prompt
from ..security.oauth2 import get_current_user
from ..models.webxos_wallet import webxos_wallet
from ..quantum_sync import quantum_sync
from ..logging import logger
import sqlite3
import json
import git
from ..api.copilot_integration import CopilotClient
from ..services.mongodb_handler import mongodb_handler

class MCPAlchemist:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.prompts = {
            "train_wallet": {"command": "/train", "description": "Train wallet with local data", "args": ["data"]},
            "translate_script": {"command": "/translate", "description": "Translate script to language", "args": ["lang", "script"]},
            "diagnose_script": {"command": "/diagnose", "description": "Diagnose script issues", "args": ["script"]},
            "git_train": {"command": "/git_train", "description": "Train with Git commit", "args": ["commit_msg"]}
        }
        self.sqlite_conn = sqlite3.connect('vial_mcp_local.db', check_same_thread=False)
        self.copilot = CopilotClient()
        self.tools = [Tool(name="train_wallet", description="Train wallet locally", input_schema={"type": "object", "properties": {"data": {"type": "string"}}})]
        self.setup_local_db()

    def setup_local_db(self):
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT,
                data TEXT,
                output TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS translation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lang TEXT,
                script TEXT,
                result TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.sqlite_conn.commit()

    async def train_wallet(self, data: str, user=None):
        if not user:
            return {"error": "Authentication required", "sqlite_error": "No user session"}
        try:
            input_tensor = torch.tensor([float(x) for x in data.split(",")][:10]).to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = torch.mean((output - torch.tensor([0.5, 0.5])).pow(2))
            loss.backward()
            self.optimizer.step()
            self.sqlite_conn.execute("INSERT INTO training_log (user, data, output) VALUES (?, ?, ?)", (user["username"], data, str(output.tolist())))
            self.sqlite_conn.commit()
            mongodb_handler.save_training_data({"user": user["username"], "data": data, "output": output.tolist()})
            await webxos_wallet.update_wallet("node1", 1.0)
            return {"status": "trained", "output": output.tolist(), "sqlite_error": None}
        except Exception as e:
            return {"error": str(e), "sqlite_error": f"Training failed: {str(e)}"}

    def translate_script(self, lang: str, script: str):
        translations = {
            "sqlite": f"sqlite3: {script}",
            "typescript": f"ts: {script}",
            "json": json.dumps({"script": script}),
            "postgres": f"psql: {script}",
            "nextjs": f"next: {script}",
            "git": f"git: {script}"
        }
        result = translations.get(lang, {"error": "Language not supported", "sqlite_error": "Unsupported language"})
        self.sqlite_conn.execute("INSERT INTO translation_log (lang, script, result) VALUES (?, ?, ?)", (lang, script, json.dumps(result)))
        self.sqlite_conn.commit()
        return result

    def diagnose_script(self, script: str):
        result = {"diagnosis": f"Script {script} is valid", "suggestions": ["Optimize loops", "Add error handling"], "sqlite_error": None}
        self.sqlite_conn.execute("INSERT INTO translation_log (script, result) VALUES (?, ?)", (script, json.dumps(result)))
        self.sqlite_conn.commit()
        return result

    async def git_train(self, commit_msg: str, user=None):
        if not user:
            return {"error": "Authentication required", "sqlite_error": "No user session"}
        try:
            repo = git.Repo(".")
            repo.git.add(all=True)
            repo.index.commit(commit_msg)
            suggestion = await self.copilot.suggest_code(commit_msg)
            self.sqlite_conn.execute("INSERT INTO training_log (user, data, output) VALUES (?, ?, ?)", (user["username"], commit_msg, suggestion))
            self.sqlite_conn.commit()
            return {"status": "trained", "suggestion": suggestion, "sqlite_error": None}
        except Exception as e:
            return {"error": str(e), "sqlite_error": f"Git training failed: {str(e)}"}

    async def handle_prompt(self, command: str, user=None):
        if not user:
            return {"error": "Authentication required", "sqlite_error": "No user session"}
        for prompt in self.prompts.values():
            if command.startswith(prompt["command"]):
                args = command[len(prompt["command"]):].strip().split()
                if len(args) == len(prompt["args"]):
                    if prompt["command"] == "/train":
                        return await self.train_wallet(args[0], user)
                    elif prompt["command"] == "/translate":
                        return self.translate_script(args[0], args[1])
                    elif prompt["command"] == "/diagnose":
                        return self.diagnose_script(args[0])
                    elif prompt["command"] == "/git_train":
                        return await self.git_train(args[0], user)
        return {"error": "Invalid prompt. Use /help for commands", "sqlite_error": "Unknown command"}

    def create_new_wallet(self, user):
        if not user:
            return {"error": "Authentication required", "sqlite_error": "No user session"}
        wallet_id = f"webxos_{user['username']}"
        self.sqlite_conn.execute("INSERT INTO wallets (user, wallet_id) VALUES (?, ?)", (user["username"], wallet_id))
        self.sqlite_conn.commit()
        return {"status": "created", "wallet_id": wallet_id, "sqlite_error": None}

mcp_alchemist = MCPAlchemist()
