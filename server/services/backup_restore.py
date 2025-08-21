from server.services.mongodb_handler import mongodb_handler
from server.logging import logger
import os
import gzip


class BackupRestore:
    def __init__(self):
        self.backup_dir = "backups"


    def backup(self):
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            backup_file = f"{self.backup_dir}/vial_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gz"
            with gzip.open(backup_file, 'wt') as f:
                for collection in mongodb_handler.db.list_collection_names():
                    for doc in mongodb_handler.db[collection].find():
                        f.write(str(doc) + '\n')
            logger.info(f"Backup created: {backup_file}")
            return {"status": "success", "file": backup_file}
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return {"status": "failed", "error": str(e)}


    def restore(self, backup_file: str):
        try:
            with gzip.open(backup_file, 'rt') as f:
                for line in f:
                    doc = eval(line)
                    collection = doc.get("collection")
                    mongodb_handler.db[collection].insert_one(doc)
            logger.info(f"Restored from: {backup_file}")
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return {"status": "failed", "error": str(e)}


backup_restore = BackupRestore()
