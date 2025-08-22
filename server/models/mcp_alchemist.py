from torch import nn


class Alchemist(nn.Module):
    def __init__(self):
        super().__init__()

    def process_prompt(self, prompt: str, context: dict):
        return {"output": f"Processed: {prompt} with {context}"}
