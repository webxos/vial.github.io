# server/config/final_config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SQLALCHEMY_DATABASE_URL: str = "sqlite:///vial.db"
    REDIS_HOST: str = "redis"
    WEBXOS_WALLET_ADDRESS: str = "e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d"
    SECRET_KEY: str = "your-secret-key-here"
    REPUTATION_LOGGING_ENABLED: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
