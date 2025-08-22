from server.services.advanced_logging import AdvancedLogger
import json


logger = AdvancedLogger()


def format_response(data: dict, status: str = "success"):
    response = {
        "status": status,
        "data": data,
        "timestamp": "2025-08-22T12:50:00"
    }
    logger.log("Response formatted", extra={"status": status, "data_keys": list(data.keys())})
    return json.dumps(response)
