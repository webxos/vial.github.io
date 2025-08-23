from typing import Dict, Any
from server.logging_config import logger
from server.services.error_logging import ErrorLogger
import smtplib
from email.mime.text import MIMEText
import os
import uuid

class NotificationService:
    def __init__(self):
        self.error_logger = ErrorLogger()
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.notify_email = os.getenv("NOTIFY_EMAIL")

    async def send_task_notification(self, task_name: str, status: str, request_id: str) -> Dict[str, Any]:
        try:
            msg = MIMEText(f"Task {task_name} completed with status: {status}")
            msg["Subject"] = f"Vial MCP Task Notification: {task_name}"
            msg["From"] = self.smtp_user
            msg["To"] = self.notify_email

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            logger.info(f"Sent notification for task {task_name}", request_id=request_id)
            return {"status": "notified", "request_id": request_id}
        except Exception as e:
            self.error_logger.log_error(f"Notification error: {str(e)}", request_id)
            logger.error(f"Notification error: {str(e)}", request_id=request_id)
            raise
