```python
import httpx
import os
from sqlalchemy.orm import Session
from ..database.gibs_models import GIBSMetadata
from ..database.base import get_db

class GIBSService:
    def __init__(self):
        self.base_url = os.getenv("GIBS_API_URL", "https://gibs.earthdata.nasa.gov")

    async def fetch_wmts_tile(self, layer: str, time: str, tile_matrix: str = "6", tile_row: str = "13", tile_col: str = "36", format_ext: str = "jpg", db: Session = None):
        url = f"{self.base_url}/wmts/epsg4326/best/{layer}/default/{time}/250m/{tile_matrix}/{tile_row}/{tile_col}.{format_ext}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            if db:
                metadata = GIBSMetadata(layer=layer, time=time, url=url, wallet_id="default")
                db.add(metadata)
                db.commit()
            return {"url": url, "layer": layer, "time": time}
```
