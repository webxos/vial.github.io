from pydantic import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    OAUTH_SECRET: str
    JWT_SECRET: str
    MONGO_URL: str = "mongodb://mongo:27017/vial"
    REDIS_URL: str = "redis://redis:6379/0"
    DATABASE_URL: str = "sqlite:///vial.db"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    NGINX_SSL_CERT: str = "/path/to/ssl/cert.pem"
    NGINX_SSL_KEY: str = "/path/to/ssl/key.pem"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings():
    return Settings()

settings = get_settings()
