from fastapi import APIRouter, Depends, HTTPException
from server.api.auth_endpoint import verify_token
import requests
import torch
import cv2
import numpy as np
import os
from typing import Dict

class NASATelescopeProcessor:
    def __init__(self):
        self.api_url = "https://api.nasa.gov/planetary/apod"
        self.api_key = os.getenv("NASA_API_KEY")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def fetch_telescope_data(self) -> Dict:
        try:
            params = {"api_key": self.api_key}
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch telescope data: {str(e)}")

    async def process_image(self, image_url: str) -> Dict:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            processed_img = torch.mean(img_tensor, dim=1, keepdim=True)
            return {"processed_image": processed_img.cpu().numpy().tolist()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

nasa_telescope = NASATelescopeProcessor()

router = APIRouter(prefix="/mcp/nasa_telescope", tags=["nasa_telescope"])

@router.get("/data")
async def get_telescope_data(token: dict = Depends(verify_token)) -> Dict:
    data = await nasa_telescope.fetch_telescope_data()
    if "url" in data:
        return await nasa_telescope.process_image(data["url"])
    return data
