from fastapi import FastAPI
from server.models.mcp_alchemist import Alchemist
from server.services.git_trainer import GitTrainer

class PromptTrainer:
    def __init__(self, app: FastAPI):
        self.app = app
        self.alchemist = Alchemist()
        self.git_trainer = GitTrainer()

    async def train(self):
        prompt = "Train model with {context}"
        context = {"data": "sample_data"}
        result = self.alchemist.process_prompt(prompt, context)
        self.git_trainer.save_training_data(result)
        return result

def setup_prompt_training(app: FastAPI):
    trainer = PromptTrainer(app)
    app.state.prompt_trainer = trainer

    @app.post("/agent/train")
    async def train_prompt_endpoint(prompt_text: str, context: dict):
        result = await app.state.prompt_trainer.train()
        return result
