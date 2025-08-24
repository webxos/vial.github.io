import re
import logging
from typing import Union
from bleach import clean
from server.config.settings import settings

logger = logging.getLogger(__name__)

def sanitize_input(input_data: Union[str, dict, list]) -> Union[str, dict, list]:
    """Sanitize input to prevent injection attacks and ensure DB isolation."""
    try:
        if isinstance(input_data, str):
            # Remove malicious SQL patterns
            sanitized = re.sub(r'(--|;|/\\*.*\\*/|union|select|insert|update|delete|drop)', '', input_data, flags=re.IGNORECASE)
            # Clean HTML/JS using bleach
            sanitized = clean(sanitized, tags=[], attributes={}, strip=True)
            logger.debug(f"Sanitized string input: {sanitized[:50]}...")
            return sanitized
        elif isinstance(input_data, dict):
            return {k: sanitize_input(v) for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [sanitize_input(item) for item in input_data]
        return input_data
    except Exception as e:
        logger.error(f"Input sanitization failed: {str(e)}")
        raise ValueError(f"Invalid input: {str(e)}")

def validate_db_instance(user_id: str) -> str:
    """Validate and isolate SQLite instance for user."""
    try:
        sanitized_user_id = sanitize_input(user_id)
        db_path = f"sqlite:///{sanitized_user_id}_vial_mcp.db"
        if not re.match(r'^[a-zA-Z0-9_-]+$', sanitized_user_id):
            raise ValueError("Invalid user ID for DB instance")
        logger.info(f"Validated DB instance for user: {sanitized_user_id}")
        return db_path
    except Exception as e:
        logger.error(f"DB instance validation failed: {str(e)}")
        raise ValueError(f"DB instance error: {str(e)}")
