import torch
import torch.nn as nn
import cv2
import numpy as np
from fastapi import HTTPException
from typing import Dict

class NASAMLModel(nn.Module):
    def __init__(self):
        super(NASAMLModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 112 * 112, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        ).to(self.device)

    def forward(self, x):
        return self.model(x)

class NASAMLProcessor:
    def __init__(self):
        self.model = NASAMLModel()
        self.model.load_state_dict(torch.load("nasa_model.pth", map_location=self.model.device))
        self.model.eval()

    async def analyze_image(self, image_data: bytes) -> Dict:
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.model.device) / 255.0
            with torch.no_grad():
                output = self.model(img_tensor)
                prediction = torch.softmax(output, dim=1)
            return {"prediction": prediction.cpu().numpy().tolist()[0]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

nasa_ml = NASAMLProcessor()
