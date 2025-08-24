from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.orm import Session
from server.config.database import Base
import logging

logging.basicConfig(level=logging.INFO, filename="logs/dao.log")

class DAOReputation(Base):
    __tablename__ = "dao_reputation"
    wallet_id = Column(String, primary_key=True)
    reputation_points = Column(Integer, default=0)
    last_updated = Column(Float, nullable=False)

class DAORepository:
    def __init__(self, db: Session):
        self.db = db

    def add_reputation(self, wallet_id: str, points: int) -> None:
        """Add reputation points to a wallet."""
        try:
            reputation = self.db.query(DAOReputation).filter(DAOReputation.wallet_id == wallet_id).first()
            if not reputation:
                reputation = DAOReputation(wallet_id=wallet_id, reputation_points=points, last_updated=time.time())
                self.db.add(reputation)
            else:
                reputation.reputation_points += points
                reputation.last_updated = time.time()
            self.db.commit()
        except Exception as e:
            logging.error(f"Failed to add reputation: {str(e)}")
            self.db.rollback()
            raise

    def get_reputation(self, wallet_id: str) -> int:
        """Get reputation points for a wallet."""
        try:
            reputation = self.db.query(DAOReputation).filter(DAOReputation.wallet_id == wallet_id).first()
            return reputation.reputation_points if reputation else 0
        except Exception as e:
            logging.error(f"Failed to get reputation: {str(e)}")
            raise
