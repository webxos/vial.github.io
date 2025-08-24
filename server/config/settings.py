import toml
from pydantic import BaseModel
import os
import logging

logging.basicConfig(level=logging.INFO, filename="logs/settings.log")

class Settings(BaseModel):
    server_host: str = "0.0.0.0"
    server_port: int = 3000
    jwt_secret: str
    anthropic_api_key: str = ""
    mistral_api_key: str = ""
    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""
    servicenow_instance: str = ""
    servicenow_user: str = ""
    servicenow_password: str = ""
    k8s_min_replicas: int = 2
    k8s_max_replicas: int = 10
    k8s_cpu_target: float = 0.7
    k8s_memory_target: str = "500Mi"
    k8s_namespace: str = "vial-mcp"
    wallet_dir: str = "~/.webxos/wallets"

    class Config:
        env_file = "mcp.toml"
        env_file_encoding = "utf-8"

def load_settings() -> Settings:
    try:
        config_path = "mcp.toml"
        if not os.path.exists(config_path):
            logging.warning("mcp.toml not found, using defaults")
            return Settings(jwt_secret="default-secret")
        config = toml.load(config_path)
        return Settings(**config)
    except Exception as e:
        logging.error(f"Failed to load settings: {str(e)}")
        raise ValueError(f"Configuration error: {str(e)}")
