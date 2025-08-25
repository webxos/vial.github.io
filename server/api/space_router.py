from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from server.agents.astronomy_agent import AstronomyAgent
from server.security.oauth2 import validate_token
from typing import Dict, Any
from datetime import datetime

router = APIRouter(prefix="/api/space")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
astronomy_agent = AstronomyAgent()

@router.get("/ephemeris/{celestial_body}")
async def get_ephemeris(
    celestial_body: str,
    lat: float = 0.0,
    lon: float = 0.0,
    date: str = None,
    token: str = Depends(oauth2_scheme)
):
    """Retrieve ephemeris data for a celestial body."""
    await validate_token(token)
    date_obj = datetime.fromisoformat(date) if date else datetime.utcnow()
    result = await astronomy_agent.calculate_ephemeris(celestial_body, lat, lon, date_obj)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@router.post("/optimal-observation")
async def optimal_observation_times(
    data: Dict[str, Any],
    token: str = Depends(oauth2_scheme)
):
    """Find optimal observation times for multiple celestial bodies."""
    await validate_token(token)
    bodies = data.get('celestial_bodies', [])
    lat = data.get('lat', 0.0)
    lon = data.get('lon', 0.0)
    date_range = data.get('date_range', (datetime.utcnow(), datetime.utcnow()))
    result = await astronomy_agent.find_optimal_observation_times(bodies, lat, lon, date_range)
    return result
