from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GITHUB_TOKEN: str
    GITHUB_USERNAME: str
    MONGO_URL: str = "mongodb://localhost:27017"
    REDIS_URL: str = "redis://localhost:6379"
    NOTIFICATION_API_URL: str = "https://api.example.com/notify"
    JWT_SECRET: str
    API_BASE_URL: str = "http://localhost:8000"
    SQLITE_DB_PATH: str = "/app/vial.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
