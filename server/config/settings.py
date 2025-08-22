from pydantic import BaseSettings


class Settings(BaseSettings):
    sql_url: str = "sqlite:///vial.db"
    mongodb_url: str = "mongodb://mongodb:27017/vial"
    redis_url: str = "redis://redis:6379"
    jwt_secret: str = "your-secret-key"
    jwt_expire_minutes: int = 30


settings = Settings()
