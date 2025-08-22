from pydantic import BaseModel


class AuthAgent(BaseModel):
    token: str
