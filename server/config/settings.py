from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Centralized configuration for Vial MCP."""
    SQLALCHEMY_DATABASE_URL: str = "sqlite:///vial_mcp.db"
    MONGO_URI: str = "mongodb://mongodb:27017/vial_mcp"
    REDIS_URL: str = "redis://redis:6379/0"
    OBS_WEBSOCKET_URL: str = "ws://localhost:4455"
    WEBXOS_WALLET_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    XAI_API_KEY: Optional[str] = None
    NASA_API_KEY: Optional[str] = None
    SERVICENOW_API_KEY: Optional[str] = None
    ALIBABA_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
