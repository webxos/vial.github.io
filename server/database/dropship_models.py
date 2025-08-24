```python
from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DropshipSimulation(Base):
    __tablename__ = "dropship_simulations"
    id = Column(Integer, primary_key=True, index=True)
    route = Column(String, index=True)
    time = Column(String)
    wallet_id = Column(String, index=True)
    data = Column(JSON)  # Stores NASA/SpaceX/Higress data

    def __repr__(self):
        return f"<DropshipSimulation(route={self.route}, wallet_id={self.wallet_id})>"
```
