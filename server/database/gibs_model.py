```python
from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from .base import Base

class GIBSMetadata(Base):
    __tablename__ = "gibs_metadata"
    id = Column(Integer, primary_key=True)
    layer = Column(String(100), nullable=False)
    time = Column(Date, nullable=False)
    url = Column(String(255), nullable=False)
    wallet_id = Column(String(64), nullable=False)

    def __repr__(self):
        return f"<GIBSMetadata(layer='{self.layer}', time='{self.time}', wallet_id='{self.wallet_id}')>"
```
