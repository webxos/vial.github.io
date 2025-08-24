```python
from fastapi import HTTPException
import os

def update_terminal(command: str) -> dict:
    """Update terminal with real-time output.
    
    Parameters:
        command: User command
    
    Returns:
        Terminal update status.
    """
    try:
        with open("public/logs/terminal.log", "a") as f:
            f.write(f"{command} at 06:42 PM EDT\n")
        return {"command": command, "status": "updated", "timestamp": "06:42 PM EDT"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(update_terminal("test command"))
```
