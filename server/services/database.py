import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect('vial_mcp.db', check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with sqlite3.connect('vial_mcp.db') as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS credentials
                        (id INTEGER PRIMARY KEY, token TEXT, expires TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS quantum_links
                        (id INTEGER PRIMARY KEY, link_id TEXT, status TEXT, time TEXT)''')
        conn.commit()
