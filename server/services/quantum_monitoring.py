from prometheus_client import Gauge, Histogram
from server.services.quantum_topology import QuantumTopologyService
import time

class QuantumMonitoringService:
    def __init__(self):
        self.quantum_service = QuantumTopologyService()
        self.topology_count = Gauge(
            'mcp_quantum_topologies_active',
            'Number of active quantum topologies'
        )
        self.topology_latency = Histogram(
            'mcp_quantum_topology_duration_seconds',
            'Quantum topology creation latency'
        )
    
    async def monitor_topology(self, qubits: int, gates: list):
        """Monitor quantum topology creation with metrics."""
        start_time = time.time()
        topology = await self.quantum_service.create_topology(qubits, gates)
        latency = time.time() - start_time
        
        self.topology_count.inc()
        self.topology_latency.observe(latency)
        return topology
    
    async def cleanup_topology(self, circuit_id: str):
        """Clean up and decrement topology count."""
        await self.quantum_service.distribute_topology(circuit_id)
        self.topology_count.dec()
