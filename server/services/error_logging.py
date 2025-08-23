from typing import List, Dict, Any
from server.logging_config import logger
import sqlite3
import uuid

class ErrorLogger:
    def __init__(self):
        self.conn = sqlite3.connect("logs.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS logs
            (id TEXT, timestamp TEXT, message TEXT, request_id TEXT)"""
        )
        self.conn.commit()

    def log_error(self, message: str, request_id: str) -> None:
        timestamp = "2025-08-23T12:01:00Z"
        log_id = str(uuid.uuid4())
        self.cursor.execute(
            "INSERT INTO logs (id, timestamp, message, request_id) VALUES (?, ?, ?, ?)",
            (log_id, timestamp, message, request_id)
        )
        self.conn.commit()
        logger.error(f"Logged error: {message}", request_id=request_id)

    def get_logs(self, request_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM logs WHERE request_id = ?", (request_id,))
        rows = self.cursor.fetchall()
        logs = [
            {"id": row[0], "timestamp": row[1], "message": row[2], "request_id": row[3]}
            for row in rows
        ]
        logger.info(f"Retrieved {len(logs)} logs for request_id {request_id}", request_id=request_id)
        return logs

    def __del__(self):
        self.conn.close()
