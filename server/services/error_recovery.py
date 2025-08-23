from fastapi import FastAPI
from server.services.vial_manager import VialManager
from server.logging import logger


def setup_error_recovery(app: FastAPI):
    vial_manager = VialManager()

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.log(f"Global error: {str(exc)}")
        # Trigger MCP Alchemist troubleshooting
        async with app.test_client() as client:
            response = await client.post("/alchemist/troubleshoot", json={"error": str(exc)})
            logger.log(f"Alchemist troubleshooting response: {response.json()}")
        return {"error": str(exc), "troubleshooting": response.json()}

    @app.post("/error/recover")
    async def recover_from_error(error_id: str):
        try:
            # Placeholder for error recovery logic
            vial_manager.restart_vial("vial1")
            logger.log(f"Recovery initiated for error: {error_id}")
            return {"status": "recovered", "error_id": error_id}
        except Exception as e:
            logger.log(f"Recovery error: {str(e)}")
            return {"error": str(e)}
