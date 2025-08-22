from server.services.advanced_logging import AdvancedLogger


logger = AdvancedLogger()


def validate_deployment(config: dict) -> dict:
    required_keys = ['name', 'components', 'connections']
    for key in required_keys:
        if key not in config:
            logger.log("Deployment validation failed",
                       extra={"error": f"Missing key: {key}"})
            return {"error": f"Missing key: {key}"}
    
    logger.log("Deployment validated",
               extra={"config_name": config.get('name')})
    return {"status": "valid"}
