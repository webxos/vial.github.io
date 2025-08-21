from server.models.mcp_alchemist import mcp_alchemist
import pytest
from pymongo import MongoClient
from server.config import get_settings

settings = get_settings()

@pytest.fixture
def mongo_client():
    client = MongoClient(settings.MONGO_URL)
    yield client
    client.close()

def test_alchemist_train(mongo_client):
    test_data = "1,2,3,4,5,6,7,8,9,10"
    result = mcp_alchemist.train(test_data)
    assert result["status"] == "training complete"
    assert "output" in result
    
    # Verify MongoDB storage
    db = mongo_client.vial
    agent = db.agents.find_one({"hash": "trained_model"})
    assert agent["data"] == test_data
    assert agent["status"] == "trained"
