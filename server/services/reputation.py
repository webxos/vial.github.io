```python
from typing import Dict
from ..database.dropship_models import DropshipSimulation
from sqlalchemy.orm import Session

class ReputationService:
    """Manages DAO wallet rewards for contributions"""
    def __init__(self, db: Session):
        self.db = db

    def track_contribution(self, wallet_id: str, action: str) -> Dict:
        """Track user contributions and assign rewards"""
        simulation = self.db.query(DropshipSimulation).filter_by(wallet_id=wallet_id).first()
        if not simulation:
            simulation = DropshipSimulation(wallet_id=wallet_id, route="moon-mars", time="2023-01-01", data={})
            self.db.add(simulation)
            self.db.commit()
        points = 10 if action == "simulation" else 5
        return {"wallet_id": wallet_id, "points": points, "md_wallet": f"{wallet_id}.md"}
```
