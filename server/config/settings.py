from pydantic import BaseSettings

class Settings(BaseSettings):
    debug: bool = True
    port: int = 8000
    sql_url: str = "sqlite:///vial.db"


settings = Settings()
