from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str = "sqlite:///data/vialmcp.db"
    github_client_id: str = ""
    github_client_secret: str = ""
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
