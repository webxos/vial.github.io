from qiskit import QuantumCircuit, execute, Aer
from server.models.base import QuantumCircuit as DBCircuit
from server.database.session import DatabaseManager
from typing import Dict, Any
import torch

class QuantumAgent:
    def __init__(self):
        self.simulator = Aer.get_backend('qasm_simulator')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_manager = DatabaseManager("sqlite:///./test.db")
    
    async def execute_quantum_circuit(self, circuit_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantum circuit as per MCP tool specification."""
        qubits = circuit_config.get('qubits', 8)
        qc = QuantumCircuit(qubits)
        
        # Apply gates from config
        for gate in circuit_config.get('gates', []):
            if gate['type'] == 'h':
                qc.h(gate['target'])
            elif gate['type'] == 'cx':
                qc.cx(gate['control'], gate['target'])
        
        # Execute circuit
        result = execute(qc, self.simulator, shots=1024).result()
        counts = result.get_counts(qc)
        
        # Save to database
        with self.db_manager.get_session() as session:
            db_circuit = DBCircuit(
                qasm_code=qc.qasm(),
                wallet_id=circuit_config.get('wallet_id')
            )
            session.add(db_circuit)
            session.commit()
            circuit_id = db_circuit.id
        
        return {
            'circuit_id': circuit_id,
            'results': counts,
            'qasm': qc.qasm()
        }
