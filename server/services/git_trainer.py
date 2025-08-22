import git

# ... (previous content assumed)

async def commit_with_mcp(self, message: str, training_data: dict):
    processed = self.alchemist.process_prompt(message, training_data)
    self.repo.git.add(all=True)
    self.repo.index.commit(f"MCP: {processed['output']}")
    return processed

import json  # Added to fix F821

async def fetch_training_data(self):
    # Assuming some JSON handling
    data = json.loads(self.repo.git.show())
    return data
