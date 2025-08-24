from pydantic_settings import BaseSettings
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)

class ScalingConfig(BaseSettings):
    """Scaling configuration for Vial MCP services."""
    QUANTUM_MIN_REPLICAS: int = 2
    QUANTUM_MAX_REPLICAS: int = 10
    QUANTUM_CPU_THRESHOLD: float = 0.7
    RAG_MIN_REPLICAS: int = 2
    RAG_MAX_REPLICAS: int = 8
    RAG_MEMORY_THRESHOLD: str = "4Gi"
    DB_CONNECTION_POOL_SIZE: int = 20
    REDIS_MAX_CONNECTIONS: int = 100
    KUBE_CONFIG_PATH: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def validate_scaling(self):
        """Validate scaling configuration."""
        if self.QUANTUM_MIN_REPLICAS > self.QUANTUM_MAX_REPLICAS:
            logger.error("Invalid quantum scaling: min_replicas > max_replicas")
            raise ValueError("Invalid quantum scaling configuration")
        if self.RAG_MIN_REPLICAS > self.RAG_MAX_REPLICAS:
            logger.error("Invalid RAG scaling: min_replicas > max_replicas")
            raise ValueError("Invalid RAG scaling configuration")
        logger.info("Scaling configuration validated successfully")

scaling_config = ScalingConfig()
scaling_config.validate_scaling()
