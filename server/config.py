import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    SQLALCHEMY_DATABASE_URI = "sqlite:///vial_mcp.db"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "your-github-token-here")
    API_PORT = 8000
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

config = Config()
