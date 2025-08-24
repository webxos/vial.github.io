```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func

Base = declarative_base()

class QuantumCircuit(Base):
    __tablename__ = "quantum_circuits"
    id = Column(Integer, primary_key=True)
    qasm_code = Column(Text, nullable=False)
    wallet_id = Column(String(64), index=True)
    created_at = Column(DateTime, server_default=func.now())

class NASADataset(Base):
    __tablename__ = "nasa_datasets"
    id = Column(Integer, primary_key=True)
    dataset_id = Column(String(128), unique=True, nullable=False)
    title = Column(String(256), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
