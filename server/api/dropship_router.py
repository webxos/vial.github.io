```python
from fastapi import APIRouter, Depends
from ..services.dropship_service import DropshipService
from ..api.middleware import oauth_middleware
from prometheus_client import Counter

router = APIRouter(prefix="/api/mcp/tools")
dropship_requests_total = Counter('mcp_dropship_requests_total', 'Total Dropship API requests')

@router.post("/dropship_data")
async def fetch_dropship_data(args: dict, request=Depends(oauth_middleware)):
    dropship_requests_total.inc()
    service = DropshipService()
    return await service.simulate_supply_chain(args, args.get("wallet_id", "default"))
```
