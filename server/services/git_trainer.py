import json


async def commit_with_mcp(self, message: str, training_data: dict):
    processed = self.alchemist.process_prompt(message, training_data)
    self.repo.git.add(all=True)
    self.repo.index.commit(f"MCP: {processed['output']}")
    return processed


async def fetch_training_data(self):
    data = json.loads(self.repo.git.show())
    return data
