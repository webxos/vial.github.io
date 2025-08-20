import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vial_mcp.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("vial_mcp")
