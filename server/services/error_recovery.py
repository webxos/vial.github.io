from fastapi import FastAPI
from server.services.advanced_logging import AdvancedLogger
from server.services.backup_restore import backup_configs


logger = AdvancedLogger()


def setup_error_recovery(app: FastAPI):
    async def recover_from_error(error: str):
        logger.log("Error recovery initiated", extra={"error": error})
        try:
            await app.state.backup_configs()
            logger.log("Error recovery completed with backup", extra={"error": error})
            return {"status": "recovered"}
        except Exception as e:
            logger.log("Error recovery failed", extra={"error": str(e)})
            return {"error": str(e)}
    
    app.state.recover_from_error = recover_from_error
    logger.log("Error recovery initialized", extra={"system": "error_recovery"})
