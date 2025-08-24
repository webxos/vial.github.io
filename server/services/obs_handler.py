```python
from typing import Dict, Optional
import obswebsocket
from obswebsocket import requests as obs_requests
from fastapi import HTTPException
import logging
from server.config.settings import settings
from server.utils.security_sanitizer import sanitize_input

logger = logging.getLogger(__name__)

class OBSHandler:
    def __init__(self):
        self.client = obswebsocket.obsws(
            host=settings.OBS_HOST,
            port=settings.OBS_PORT,
            password=settings.OBS_PASSWORD
        )
        try:
            self.client.connect()
        except Exception as e:
            logger.error(f"OBS connection failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def init_scene(self, scene_name: str) -> Dict:
        """Initialize an OBS scene."""
        try:
            sanitized_scene = sanitize_input(scene_name)
            self.client.call(obs_requests.CreateScene(sanitized_scene))
            logger.info(f"OBS scene initialized: {sanitized_scene}")
            return {"status": "success", "scene": sanitized_scene}
        except Exception as e:
            logger.error(f"OBS scene initialization failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def disconnect(self):
        """Disconnect from OBS WebSocket."""
        try:
            self.client.disconnect()
            logger.info("OBS WebSocket disconnected")
        except Exception as e:
            logger.error(f"OBS disconnection failed: {str(e)}")
```
