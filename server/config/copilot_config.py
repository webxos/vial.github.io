from pydantic import BaseSettings
from server.config import Settings

class CopilotSettings(Settings):
    GITHUB_TOKEN: str = "placeholder_token"
    COPILOT_MAX_SNIPPETS: int = 3
    COPILOT_QUERY_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_copilot_settings():
    return CopilotSettings()

copilot_settings = get_copilot_settings()
