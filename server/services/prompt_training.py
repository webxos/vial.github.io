from server.models.mcp_alchemist import Alchemist
import torch.nn as nn


def setup_prompt_training(app):
    alchemist = Alchemist()
    for i in range(1, 5):
        vial_id = f"vial{i}"
        model = nn.Linear(10, 1) if i == 1 else nn.Linear(20, 2)
        alchemist.train(vial_id, model)
