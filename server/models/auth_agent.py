from pydantic import BaseModel


class AuthAgent(BaseModel):
    user_id: str
    token: str

    def process_user(self, user_data: dict):
        return {"user_id": user_data.get("id"), "username": user_data.get("login")}
