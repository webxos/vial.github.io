import sqlite3
from functools import wraps
from fastapi import HTTPException

def handle_sqlite_error(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        db = kwargs.get('db')
        try:
            return await func(*args, **kwargs)
        except sqlite3.Error as e:
            if db:
                db.rollback()
            raise HTTPException(status_code=500, detail=f"SQLite Error: {str(e)}")
        finally:
            if db:
                db.close()
    return wrapper
