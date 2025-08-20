from fastapi import APIRouter
from ..quantum.network import quantum_network

router = APIRouter()

@router.post("/quantum_establish_link")
async def establish_quantum_link(node_a: str, node_b: str):
    return await quantum_network.establish_quantum_link(node_a, node_b)
