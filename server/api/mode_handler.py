```python
import agora
import camel.types
from fastapi import HTTPException

toolformer = agora.toolformers.CamelToolformer(camel.types.ModelPlatformType.OPENAI, camel.types.ModelType.GPT_4O)
receiver = agora.Receiver.make_default(toolformer)

@receiver.task()
def handle_mode(mode: str) -> dict:
    """Handle mode switching with Agora.
    
    Parameters:
        mode: Selected mode (SVG, LAUNCH, SWARM, GALAXYCRAFT)
    
    Returns:
        Mode status dictionary.
    """
    modes = {"SVG": "active", "LAUNCH": "active", "SWARM": "active", "GALAXYCRAFT": "active"}
    if mode not in modes:
        raise HTTPException(status_code=400, detail="Invalid mode")
    return {"mode": mode, "status": modes[mode], "timestamp": "06:35 PM EDT"}

if __name__ == "__main__":
    print(handle_mode("SVG"))
```
