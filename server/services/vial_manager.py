from typing import Dict
from server.logging import logger


class VialManager:
    def __init__(self):
        self.agents = {
            f"vial{i}": {"status": "running", "model": None} for i in range(1, 5)
        }

    def get_vial_status(self, vial_id: str) -> Dict:
        try:
            if vial_id not in self.agents:
                raise ValueError(f"Vial {vial_id} not found")
            status = self.agents[vial_id]["status"]
            logger.log(f"Vial status checked: {vial_id} - {status}")
            return {"vial_id": vial_id, "status": status}
        except Exception as e:
            logger.log(f"Vial status error: {str(e)}")
            return {"error": str(e)}

    def restart_vial(self, vial_id: str) -> Dict:
        try:
            if vial_id not in self.agents:
                raise ValueError(f"Vial {vial_id} not found")
            self.agents[vial_id]["status"] = "restarting"
            # Simulate restart
            self.agents[vial_id]["status"] = "running"
            logger.log(f"Vial restarted: {vial_id}")
            return {"status": "restarted", "vial_id": vial_id}
        except Exception as e:
            logger.log(f"Vial restart error: {str(e)}")
            return {"error": str(e)}

    def train_vial(self, vial_id: str, prompt: str) -> Dict:
        try:
            if vial_id not in self.agents:
                raise ValueError(f"Vial {vial_id} not found")
            # Placeholder for training logic
            logger.log(f"Vial training initiated: {vial_id} with prompt: {prompt}")
            return {"status": "trained", "vial_id": vial_id}
        except Exception as e:
            logger.log(f"Vial training error: {str(e)}")
            return {"error": str(e)}
