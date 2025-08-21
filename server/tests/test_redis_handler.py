from server.services.redis_handler import redis_handler
from unittest.mock import patch


def test_redis_get_set(mock_redis):
    mock_redis.return_value.get.return_value = "1"
    mock_redis.return_value.setex.return_value = None
    result = redis_handler.get("test_key")
    assert result == "1"
    redis_handler.setex("test_key", 60, "2")
    mock_redis.return_value.setex.assert_called_with("test_key", 60, "2")


def test_redis_incr(mock_redis):
    mock_redis.return_value.incr.return_value = 2
    result = redis_handler.incr("test_key")
    assert result == 2
    mock_redis.return_value.incr.assert_called_with("test_key")
