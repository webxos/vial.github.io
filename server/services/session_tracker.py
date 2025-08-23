from typing import Dict, Any
from server.services.memory_manager import MemoryManager
from server.logging_config import logger
import uuid

class SessionTracker:
    def __init__(self):
        self.memory_manager = MemoryManager()

    async def track_activity(self, token: str, activity: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        try:
            session = await self.memory_manager.get_session(token, request_id)
            activities = session.get("activities", [])
            activities.append({
                "action": activity.get("action"),
                "timestamp": "2025-08-23T04:02:00Z",
                "details": activity.get("details", {})
            })
            session_data = {
                "menu_info": session.get("menu_info", {}),
                "build_progress": session.get("build_progress", []),
                "quantum_logic": session.get("quantum_logic", {}),
                "task_memory": session.get("task_memory", []),
                "activities": activities
            }
            await self.memory_manager.save_session(token, session_data, request_id)
            logger.info(f"Tracked activity for token {token}", request_id=request_id)
            return {"status": "tracked", "request_id": request_id}
        except Exception as e:
            logger.error(f"Activity tracking error: {str(e)}", request_id=request_id)
            raise
