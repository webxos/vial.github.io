from fastapi import APIRouter, Depends
from server.services.vial_manager import VialManager
from server.services.database import get_db
from server.models.webxos_wallet import WalletModel
from server.logging import logger
import uuid
from datetime import datetime

router = APIRouter()


@router.post("/export")
async def export_wallet(user_id: str, db=Depends(get_db)):
    wallet = db.query(WalletModel).filter(WalletModel.user_id == user_id).first()
    if not wallet:
        wallet = WalletModel(
            user_id=user_id,
            balance=72017.0,
            network_id=str(uuid.uuid4())
        )
        db.add(wallet)
        db.commit()
    vial_manager = VialManager()
    export_data = {
        "network_id": wallet.network_id,
        "session_start": datetime.utcnow().isoformat() + "Z",
        "reputation": 1229811727985,
        "wallet": {
            "key": str(uuid.uuid4()),
            "balance": wallet.balance,
            "address": str(uuid.uuid4()),
            "hash": "042e2b6c16cc0471417e0bca0161be72258214efcf46953a63c6343b187887ce"
        },
        "vials": [
            {
                "name": f"vial{i}",
                "status": "running",
                "language": "Python",
                "balance": 18004.25,
                "address": str(uuid.uuid4()),
                "hash": "042e2b6c16cc0471417e0bca0161be72258214efcf46953a63c6343b187887ce",
                "svg_diagram": generate_svg_diagram(f"vial{i}")
            }
            for i in range(1, 5)
        ]
    }
    logger.log(f"Exported wallet with SVG for user: {user_id}")
    return export_data


def generate_svg_diagram(vial_id: str) -> str:
    return f"""<svg width="200" height="200">
        <rect x="10" y="10" width="180" height="180" fill="#3498db"/>
        <text x="50" y="100" fill="white">{vial_id}</text>
    </svg>"""
