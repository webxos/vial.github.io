from celery import Celery
from celery.schedules import crontab
from server.database.models import QuantumCircuit, VideoFrame, Wallet
from server.services.quantum_service import QuantumService
import redis.asyncio as redis
from pymongo import MongoClient
import asyncio

app = Celery('webxos_mcp', broker='redis://redis:6379/0', backend='redis://redis:6379/0')
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    beat_schedule={
        'health-check-every-minute': {
            'task': 'server.tasks.celery_app.health_check',
            'schedule': crontab(minute='*/1'),
        },
    },
)

quantum_service = QuantumService()
redis_client = redis.from_url('redis://redis:6379/0')
mongo_client = MongoClient('mongodb://mongo:27017')

@app.task
async def health_check():
    """Perform recursive system health checks."""
    try:
        # Check Redis
        await redis_client.ping()
        
        # Check MongoDB
        mongo_client.admin.command('ping')
        
        # Check database consistency
        with quantum_service.db_manager.get_session() as session:
            circuit_count = session.query(QuantumCircuit).count()
            wallet_count = session.query(Wallet).count()
        
        # Verify quantum model status
        model_status = quantum_service.model.training
        return {
            'status': 'healthy',
            'services': {'redis': 'up', 'mongodb': 'up', 'sqlalchemy': 'up'},
            'metrics': {'circuits': circuit_count, 'wallets': wallet_count, 'model': model_status}
        }
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

@app.task
async def process_quantum_circuit(circuit_id: str):
    """Process quantum circuit asynchronously."""
    try:
        result = await quantum_service.sync_circuit(circuit_id)
        return {'status': 'success', 'circuit_id': circuit_id, 'result': result}
    except Exception as e:
        return {'status': 'error', 'circuit_id': circuit_id, 'error': str(e)}
