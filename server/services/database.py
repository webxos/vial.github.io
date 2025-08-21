from server.config import get_settings
from server.logging import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class Database:
    def __init__(self):
        settings = get_settings()
        self.engine = create_engine(settings.MONGO_URL)
        self.Session = sessionmaker(bind=self.engine)


    def get_session(self):
        return self.Session()


    def execute_query(self, query: str):
        try:
            session = self.Session()
            result = session.execute(query)
            session.commit()
            return result.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise ValueError(f"Query execution failed: {str(e)}")


    def close(self):
        self.engine.dispose()


database = Database()
