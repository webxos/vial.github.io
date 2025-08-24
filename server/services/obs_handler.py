import asyncio
from obswebsocket import obsws, requests
import logging

logging.basicConfig(level=logging.INFO, filename="logs/obs_handler.log")

async def init_obs_scene(scene_name: str, host: str, port: int, password: str) -> str:
    """Initialize an OBS scene with WebSocket."""
    try:
        ws = obsws(host, port, password)
        await asyncio.to_thread(ws.connect)
        
        # Validate scene
        response = await asyncio.to_thread(ws.call, requests.GetSceneList())
        scenes = [scene["sceneName"] for scene in response.datain["scenes"]]
        if scene_name not in scenes:
            await asyncio.to_thread(ws.call, requests.CreateScene(scene_name))
        
        # Set current scene
        await asyncio.to_thread(ws.call, requests.SetCurrentProgramScene(scene_name))
        await asyncio.to_thread(ws.disconnect)
        
        logging.info(f"Initialized OBS scene: {scene_name}")
        return scene_name
    except Exception as e:
        logging.error(f"OBS connection error: {str(e)}")
        raise ValueError(f"Failed to initialize OBS scene: {str(e)}")
