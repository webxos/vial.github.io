from server.logging import logger


def get_config():
    try:
        return {"model": "claude-3-opus", "enabled": True}
    except Exception as e:
        logger.error(f"Failed to load copilot config: {str(e)}")
        raise ValueError(f"Copilot config load failed: {str(e)}")


def generate_suggestions(query: dict, config: dict):
    return ["suggestion1", "suggestion2"]
