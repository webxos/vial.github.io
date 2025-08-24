```python
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sender_agent import get_mode_status
import shutil

app = FastAPI()

@app.get("/api/mcp/status")
async def get_status():
    try:
        return {"message": "Status OK", "timestamp": "06:16 PM EDT"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mode")
async def switch_mode(mode: dict):
    try:
        return {"status": "active", "timestamp": "06:16 PM EDT"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agora/send")
async def send_agora(mode: dict):
    try:
        response = get_mode_status(mode.get("mode", "unknown"))
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/svg/transpile")
async def transpile_svg(svg: UploadFile = File(...)):
    try:
        with open(f"public/uploads/{svg.filename}", "wb") as buffer:
            shutil.copyfileobj(svg.file, buffer)
        return {"status": "transpiled", "filename": svg.filename, "timestamp": "06:16 PM EDT"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
