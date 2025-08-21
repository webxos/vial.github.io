from server.services.mongodb_handler import mongodb_handler
from server.config import get_settings
from server.logging import logger
import os
import subprocess
import datetime

class BackupRestore:
    def __init__(self):
        self.settings = get_settings()
        self.db = mongodb_handler.db

    def backup(self, backup_dir: str = "backups"):
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{backup_dir}/vial_backup_{timestamp}.gz"
        try:
            subprocess.run(
                ["mongodump", "--uri", self.settings.MONGO_URL, "--archive", backup_file],
                check=True
            )
            logger.info(f"Backup created: {backup_file}")
            return {"status": "backup created", "file": backup_file}
        except subprocess.CalledProcessError as e:
            logger.error(f"Backup failed: {str(e)}")
            raise ValueError(f"Backup failed: {str(e)}")

    def restore(self, backup_file: str):
        try:
            subprocess.run(
                ["mongorestore", "--uri", self.settings.MONGO_URL, "--archive", backup_file, "--drop"],
                check=True
            )
            logger.info(f"Database restored from: {backup_file}")
            return {"status": "restore complete"}
        except subprocess.CalledProcessError as e:
            logger.error(f"Restore failed: {str(e)}")
            raise ValueError(f"Restore failed: {str(e)}")

backup_restore = BackupRestore()
