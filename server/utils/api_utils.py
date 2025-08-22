from server.services.advanced_logging import AdvancedLogger
from pydantic import BaseModel
import json


logger = AdvancedLogger()


def validate_payload(payload: dict, model: BaseModel):
    try:
        model(**payload)
        logger.log("Payload validated", extra={"payload_keys": list(payload.keys())})
        return {"status": "valid"}
    except Exception as e:
        logger.log("Payload validation failed", extra={"error": str(e)})
        return {"error": str(e)}


def serialize_response(data: dict):
    serialized = json.dumps(data, default=str)
    logger.log("Response serialized", extra={"data_size": len(serialized)})
    return serialized
