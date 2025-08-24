```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/mcp/status")
async def get_status():
    try:
        # Simulated status check
        return {"message": "Status OK", "timestamp": "05:47 PM EDT"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
