from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from server.models.mcp_alchemist import Alchemist
from server.services.git_trainer import GitTrainer


class PromptTrainer:
    def __init__(self, app: FastAPI):
        self.app = app
        self.alchemist = Alchemist()
        self.git_trainer = GitTrainer()

    async def train_prompt(self, prompt_text: str, context: dict):
        prompt = PromptTemplate(input_variables=["context"], template=prompt_text)
        trained_output = await self.alchemist.process_prompt(prompt, context)
        await self.git_trainer.save_training_data(trained_output)
        return trained_output


def setup_prompt_training(app: FastAPI):
    trainer = PromptTrainer(app)
    app.state.prompt_training = trainer

    @app.post("/agent/train")
    async def train_prompt_endpoint(prompt_text: str, context: dict):
        return await app.state.prompt_training.train_prompt(prompt_text, context)
