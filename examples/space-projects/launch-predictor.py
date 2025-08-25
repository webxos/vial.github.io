import asyncio
from server.services.spacex_service import SpaceXService
from datetime import datetime

async def predict_next_launch():
    spacex = SpaceXService()
    launches = await spacex.get_launches(limit=10)
    upcoming = [launch for launch in launches if datetime.fromisoformat(launch['date_utc'].replace('Z', '+00:00')) > datetime.utcnow()]
    if upcoming:
        next_launch = min(upcoming, key=lambda x: datetime.fromisoformat(x['date_utc'].replace('Z', '+00:00')))
        print(f"Next launch: {next_launch['name']} at {next_launch['date_utc']}")
    else:
        print("No upcoming launches found")

if __name__ == "__main__":
    asyncio.run(predict_next_launch())
