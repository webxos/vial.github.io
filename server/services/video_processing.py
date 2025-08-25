from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
import obswebsocket
import obswebsocket.requests
from obswebsocket import obsws
import cv2
import numpy as np
import torch
import os
from fastapi.responses import StreamingResponse
from typing import Dict

class VideoProcessingService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs = obsws(os.getenv("OBS_HOST", "localhost"), 4455, os.getenv("OBS_PASSWORD", "password"))
        self.obs.connect()
        self.cap = cv2.VideoCapture(0)  # Webcam as example

    async def stream_video(self) -> StreamingResponse:
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    raise HTTPException(status_code=500, detail="Failed to capture video")
                # CUDA-accelerated processing
                frame_tensor = torch.from_numpy(frame).to(self.device).float() / 255.0
                processed_frame = cv2.cvtColor(frame_tensor.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Video streaming failed: {str(e)}")
        finally:
            self.cap.release()
            self.obs.disconnect()

    async def start_obs_stream(self, scene: str):
        try:
            await self.obs.call(obswebsocket.requests.SetCurrentScene(scene))
            await self.obs.call(obswebsocket.requests.StartStreaming())
            return {"message": f"Streaming started on scene {scene}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OBS streaming failed: {str(e)}")

video_processing = VideoProcessingService()

router = APIRouter(prefix="/mcp/video", tags=["video"])

@router.get("/stream")
async def video_stream(token: dict = Depends(verify_token)):
    return StreamingResponse(video_processing.stream_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.post("/obs/start")
async def start_obs(scene: str, token: dict = Depends(verify_token)) -> Dict:
    return await video_processing.start_obs_stream(scene)
