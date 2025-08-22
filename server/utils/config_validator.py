from pydantic import BaseModel
from server.services.advanced_logging import AdvancedLogger


logger = AdvancedLogger()


class ConfigSchema(BaseModel):
    name: str
    components: list
    connections: list


def validate_config(config: dict):
    try:
        ConfigSchema(**config)
        logger.log("Configuration validated", extra={"config_name": config.get("name")})
        return {"status": "valid"}
    except Exception as e:
        logger.log("Configuration validation failed", extra={"error": str(e)})
        return {"error": str(e)}
