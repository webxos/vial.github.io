# server/utils/response_formatter.py
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def format_response(data: Dict[str, Any], status: str = "success") -> Dict[str, Any]:
    """Format API response with wallet and reputation data."""
    try:
        formatted = {
            "status": status,
            "data": data,
            "timestamp": "2025-08-22T15:45:00Z"
        }
        logger.info(
            f"Response formatted: status={status}, "
            f"data_keys={list(data.keys())}"
        )
        return formatted
    except Exception as e:
        logger.error(f"Response formatting error: {str(e)}")
        return {"status": "error", "error": str(e)}
