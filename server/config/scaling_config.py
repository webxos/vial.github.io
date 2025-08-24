```python
from typing import Dict, Optional
from pydantic import BaseModel, validator
import logging
from server.config.settings import settings

logger = logging.getLogger(__name__)

class ScalingConfig(BaseModel):
    min_replicas: int = 2
    max_replicas: int = 10
    cpu_target: float = 0.7
    memory_target: str = "500Mi"
    namespace: str = "vial-mcp"

    @validator("min_replicas", "max_replicas")
    def validate_replicas(cls, v: int, values: Dict) -> int:
        if "min_replicas" in values and "max_replicas" in values:
            if v < values.get("min_replicas", v) and v == values["max_replicas"]:
                raise ValueError("max_replicas must be >= min_replicas")
        if v < 1:
            raise ValueError("Replicas must be >= 1")
        return v

    @validator("cpu_target")
    def validate_cpu(cls, v: float) -> float:
        if not 0.1 <= v <= 0.9:
            raise ValueError("cpu_target must be between 0.1 and 0.9")
        return v

    @validator("memory_target")
    def validate_memory(cls, v: str) -> str:
        if not v.endswith(("Mi", "Gi")) or not v[:-2].isdigit():
            raise ValueError("memory_target must be in Mi or Gi format")
        return v

class ScalingManager:
    def __init__(self):
        self.config = ScalingConfig(
            min_replicas=settings.K8S_MIN_REPLICAS,
            max_replicas=settings.K8S_MAX_REPLICAS,
            cpu_target=settings.K8S_CPU_TARGET,
            memory_target=settings.K8S_MEMORY_TARGET,
            namespace=settings.K8S_NAMESPACE
        )

    async def apply_scaling(self) -> Dict:
        """Apply Kubernetes scaling configuration."""
        try:
            scaling_spec = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {"name": "vial-mcp-hpa", "namespace": self.config.namespace},
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": "vial-mcp"
                    },
                    "minReplicas": self.config.min_replicas,
                    "maxReplicas": self.config.max_replicas,
                    "metrics": [
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "cpu",
                                "target": {"type": "Utilization", "averageUtilization": int(self.config.cpu_target * 100)}
                            }
                        },
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "memory",
                                "target": {"type": "AverageValue", "averageValue": self.config.memory_target}
                            }
                        }
                    ]
                }
            }
            logger.info(f"Applying scaling config: {self.config.dict()}")
            return scaling_spec
        except Exception as e:
            logger.error(f"Scaling configuration failed: {str(e)}")
            raise
```
