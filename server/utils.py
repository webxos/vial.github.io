import json
from server.logging import logger


def load_config(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {file_path}: {str(e)}")
        raise ValueError(f"Config load failed: {str(e)}")


def save_config(file_path: str, config: dict):
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {file_path}: {str(e)}")
        raise ValueError(f"Config save failed: {str(e)}")
