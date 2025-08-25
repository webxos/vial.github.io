import torch
from fastapi import APIRouter, Depends
from server.security.oauth2 import validate_token
from server.video.video_processor import VideoProcessor

router = APIRouter(prefix="/cuda/test/pytorch")

class PyTorchCudaTest:
    def __init__(self):
        self.processor = VideoProcessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def run_test(self, token: str = Depends(validate_token)):
        if self.device.type == "cuda":
            model = self.processor.load_model().to(self.device)
            input_tensor = torch.randn(1, 3, 1080, 1920).to(self.device)
            output = model(input_tensor)
            return {"status": "success", "device": self.device.type}
        return {"status": "failed", "device": self.device.type}
