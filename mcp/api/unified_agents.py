from fastapi import APIRouter, Depends
from pydantic import BaseModel
import json
import os
from langchain.llms import OpenAI
from sqlalchemy import create_engine
import pymongo

router = APIRouter()
SECRET_KEY = os.getenv("NEXTAUTH_SECRET", "default-secret")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/vialmcp")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
engine = create_engine(DATABASE_URL)
mongo_client = pymongo.MongoClient(MONGO_URL)
db = mongo_client["vialmcp"]
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key"))

class DataRequest(BaseModel):
    type: str
    content: dict

def translate_data(data: dict, target_type: str) -> dict:
    if target_type == "json":
        return json.dumps(data)
    elif target_type == "postgres":
        with engine.connect() as conn:
            conn.execute("INSERT INTO data (content) VALUES (:content)", {"content": json.dumps(data)})
            conn.commit()
        return {"status": "saved"}
    elif target_type == "mongodb":
        db.data.insert_one(data)
        return {"status": "saved"}
    return data

@router.post("/process")
async def process_data(request: DataRequest, current_user: dict = Depends(lambda: {"username": "test"})):
    return translate_data(request.content, request.type)
