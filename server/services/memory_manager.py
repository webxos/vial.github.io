from typing import Dict, Any, List
from pymongo import MongoClient
from server.services.error_logging import ErrorLogger
from server.logging_config import logger
import os
import uuid

class MemoryManager:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.db = self.mongo_client["vial_mcp"]
        self.error_logger = ErrorLogger()

    async def save_session(self, token: str, session_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        try:
            session = {
                "token": token,
                "menu_info": session_data.get("menu_info", {}),
                "build_progress": session_data.get("build_progress", []),
                "quantum_logic": session_data.get("quantum_logic", {}),
                "task_memory": session_data.get("task_memory", []),
                "timestamp": "2025-08-23T03:41:00Z",
                "request_id": request_id
            }
            self.db.sessions.update_one({"token": token}, {"$set": session}, upsert=True)
            logger.info(f"Saved session for token {token}", request_id=request_id)
            return {"status": "saved", "request_id": request_id}
        except Exception as e:
            self.error_logger.log_error(f"Session save error: {str(e)}", request_id)
            logger.error(f"Session save error: {str(e)}", request_id=request_id)
            raise

    async def get_session(self, token: str, request_id: str) -> Dict[str, Any]:
        try:
            session = self.db.sessions.find_one({"token": token})
            if session:
                logger.info(f"Retrieved session for token {token}", request_id=request_id)
                return session
            logger.info(f"No session found for token {token}", request_id=request_id)
            return {}
        except Exception as e:
            self.error_logger.log_error(f"Session retrieval error: {str(e)}", request_id)
            logger.error(f"Session retrieval error: {str(e)}", request_id=request_id)
            raise

    async def reset_session(self, token: str, request_id: str) -> Dict[str, Any]:
        try:
            self.db.sessions.delete_one({"token": token})
            logger.info(f"Reset session for token {token}", request_id=request_id)
            return {"status": "reset", "request_id": request_id}
        except Exception as e:
            self.error_logger.log_error(f"Session reset error: {str(e)}", request_id)
            logger.error(f"Session reset error: {str(e)}", request_id=request_id)
            raise

    async def save_task_relationship(self, task_name: str, related_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        try:
            relationship = {
                "task_name": task_name,
                "quantum_logic": related_data.get("quantum_logic", {"qubits": 0, "gates": []}),
                "training_data": related_data.get("training_data", {}),
                "related_tasks": related_data.get("related_tasks", []),
                "timestamp": "2025-08-23T03:41:00Z",
                "request_id": request_id
            }
            self.db.task_relationships.insert_one(relationship)
            logger.info(f"Saved task relationship for {task_name}", request_id=request_id)
            return {"status": "saved", "request_id": request_id}
        except Exception as e:
            self.error_logger.log_error(f"Task relationship save error: {str(e)}", request_id)
            logger.error(f"Task relationship save error: {str(e)}", request_id=request_id)
            raise

    async def get_quantum_logic(self, task_name: str, request_id: str) -> Dict[str, Any]:
        try:
            relationship = self.db.task_relationships.find_one({"task_name": task_name})
            if relationship and relationship.get("quantum_logic"):
                logger.info(f"Retrieved quantum logic for {task_name}", request_id=request_id)
                return relationship["quantum_logic"]
            logger.info(f"No quantum logic found for {task_name}", request_id=request_id)
            return {"qubits": 0, "gates": []}
        except Exception as e:
            self.error_logger.log_error(f"Quantum logic retrieval error: {str(e)}", request_id)
            logger.error(f"Quantum logic retrieval error: {str(e)}", request_id=request_id)
            raise
