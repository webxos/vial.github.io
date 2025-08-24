```python
import agora
import camel.types
from agora_utils import extract_metadata

toolformer = agora.toolformers.CamelToolformer(camel.types.ModelPlatformType.OPENAI, camel.types.ModelType.GPT_4O)

def mode_handler(mode_data: dict) -> dict:
    return {"mode": mode_data.get("mode", "unknown"), "status": "processed", "timestamp": "06:16 PM EDT"}

receiver = agora.Receiver.make_default(toolformer, tools=[mode_handler])
server = agora.ReceiverServer(receiver)

if __name__ == "__main__":
    server.run(port=5000)
```
