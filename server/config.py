import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/vial_mcp")
    SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL", "sqlite:///vial.db")
    GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
    GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/callback")
    VPS_IP = os.getenv("VPS_IP")
    WEB3_PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL", "https://mainnet.infura.io/v3/your_infura_project_id")
    JWT_SECRET = os.getenv("JWT_SECRET", "your_jwt_secret")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    PORT = int(os.getenv("PORT", 8000))

config = Config()
