from pydantic import BaseSettings
from server.logging import logger


class Settings(BaseSettings):
    GITHUB_TOKEN: str
    GITHUB_USERNAME: str
    MONGO_URL: str
    REDIS_URL: str
    NOTIFICATION_API_URL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings():
    try:
        return Settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {str(e)}")
        raise ValueError(f"Settings load failed: {str(e)}")


settings = get_settings()
