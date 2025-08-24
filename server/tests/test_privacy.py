```python
import pytest
from ..privacy_engine import PlanetaryPrivacyEngine

@pytest.fixture
def privacy_engine():
    return PlanetaryPrivacyEngine()

def test_privacy_engine_initialization(privacy_engine):
    assert privacy_engine.automated_compliance is True
    assert isinstance(privacy_engine.regional_rules, dict)

def test_process_data(privacy_engine):
    data = {"user_id": "test123", "data": "sensitive"}
    result = privacy_engine.process_data(data, "US")
    assert "processed_data" in result
    assert result["compliance_status"] == "GDPR++"
```
