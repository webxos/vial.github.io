import torch
import torch.nn as nn
from server.models.dao_repository import DAORepository
from server.models.ai_training_sessions import AITrainingSessionRepository
from sqlalchemy.orm import Session
import yaml
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, filename="logs/webxos_wallet.log")

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input: task complexity, quality, impact
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output: reward score
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class WebXOSWallet:
    def __init__(self, db: Session, config_path: str = "config/wallet_config.yaml"):
        self.db = db
        self.dao_repo = DAORepository(db)
        self.training_repo = AITrainingSessionRepository(db)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_model = RewardModel().to(self.device)
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str) -> Dict:
        """Load wallet configuration from YAML."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded wallet config from {config_path}")
            return config or {"reward_base": 10, "max_reward": 100}
        except Exception as e:
            self.logger.error(f"Error loading wallet config: {str(e)}")
            return {"reward_base": 10, "max_reward": 100}

    async def calculate_reward(self, wallet_id: str, task: Dict, training_session_id: Optional[str] = None) -> float:
        """Calculate reward using PyTorch model."""
        try:
            # Extract task metrics
            complexity = task.get("complexity", 1.0)
            quality = task.get("quality", 0.8)
            impact = task.get("impact", 0.5)
            
            # Prepare input tensor
            inputs = torch.tensor([complexity, quality, impact], dtype=torch.float32).to(self.device)
            
            # Calculate reward
            with torch.no_grad():
                reward = self.reward_model(inputs).item()
            
            # Apply config constraints
            reward = min(max(reward, 0), self.config["max_reward"]) + self.config["reward_base"]
            
            # Update DAO reputation
            self.dao_repo.add_reputation(wallet_id, points=int(reward))
            
            # Log to training sessions if applicable
            if training_session_id:
                self.training_repo.add_session(
                    model_type="reward_model",
                    video_source=task.get("video_source", "unknown"),
                    training_metrics={"reward": reward, "wallet_id": wallet_id}
                )
            
            self.logger.info(f"Calculated reward {reward} for wallet {wallet_id}")
            return reward
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            raise

# Example wallet_config.yaml:
# reward_base: 10
# max_reward: 100
# complexity_weight: 0.4
# quality_weight: 0.4
# impact_weight: 0.2
