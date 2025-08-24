```python
import os

class Config:
    def __init__(self):
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", 8000))
        self.nasa_api_key = os.getenv("NASA_API_KEY")
        self.spacex_api_url = os.getenv("SPACEX_API_URL", "https://api.spacexdata.com/v4")
        self.gibs_api_url = os.getenv("GIBS_API_URL", "https://gibs.earthdata.nasa.gov")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", 1234))
```
