```python
import pytest
from unittest.mock import patch
from server.config.scaling_config import ScalingConfig, ScalingManager
from pydantic import ValidationError

@pytest.mark.asyncio
async def test_scaling_config_valid():
    """Test valid scaling configuration."""
    config = ScalingConfig(min_replicas=2, max_replicas=10, cpu_target=0.7, memory_target="500Mi", namespace="vial-mcp")
    manager = ScalingManager()
    with patch("server.config.settings.settings", return_value={"K8S_MIN_REPLICAS": 2, "K8S_MAX_REPLICAS": 10, "K8S_CPU_TARGET": 0.7, "K8S_MEMORY_TARGET": "500Mi", "K8S_NAMESPACE": "vial-mcp"}):
        result = await manager.apply_scaling()
        assert result["spec"]["minReplicas"] == 2
        assert result["spec"]["maxReplicas"] == 10
        assert result["spec"]["metrics"][0]["resource"]["name"] == "cpu"

@pytest.mark.asyncio
async def test_scaling_config_invalid_replicas():
    """Test invalid replicas configuration."""
    with pytest.raises(ValidationError) as exc:
        ScalingConfig(min_replicas=10, max_replicas=5)
    assert "max_replicas must be >= min_replicas" in str(exc.value)

@pytest.mark.asyncio
async def test_scaling_config_invalid_cpu():
    """Test invalid CPU target configuration."""
    with pytest.raises(ValidationError) as exc:
        ScalingConfig(cpu_target=1.0)
    assert "cpu_target must be between 0.1 and 0.9" in str(exc.value)

@pytest.mark.asyncio
async def test_scaling_config_invalid_memory():
    """Test invalid memory target configuration."""
    with pytest.raises(ValidationError) as exc:
        ScalingConfig(memory_target="invalid")
    assert "memory_target must be in Mi or Gi format" in str(exc.value)
```
