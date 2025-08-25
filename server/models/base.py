from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()

class QuantumCircuit(Base):
    __tablename__ = "quantum_circuits"
    
    id = Column(Integer, primary_key=True)
    qasm_code = Column(Text, nullable=False)
    wallet_id = Column(String(64), index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class VideoFrame(Base):
    __tablename__ = "video_frames"
    
    id = Column(Integer, primary_key=True)
    svg_path = Column(String(255), nullable=False)
    duration = Column(Integer, nullable=False)
    output_path = Column(String(255), nullable=False)
    wallet_id = Column(String(64), index=True)
    created_at = Column(DateTime, server_default=func.now())

class Wallet(Base):
    __tablename__ = "wallets"
    
    id = Column(String(64), primary_key=True)
    balance = Column(Float, default=0.0)
    address = Column(String(64), unique=True)
    hash = Column(String(64), unique=True)
    created_at = Column(DateTime, server_default=func.now())
