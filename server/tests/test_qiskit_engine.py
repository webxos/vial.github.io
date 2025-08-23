import pytest
from server.quantum.qiskit_engine import QiskitEngine
from server.models.visual_components import ComponentModel, ComponentType, Position3D


@pytest.fixture
def qiskit_engine():
    return QiskitEngine()


def test_build_circuit_with_valid_components(qiskit_engine):
    components = [
        ComponentModel(
            id="comp1",
            type=ComponentType.AGENT,
            title="Agent 1",
            position=Position3D(x=0, y=0, z=0),
            config={"vial_id": "vial1"},
            connections=[],
            svg_style="default"
        ),
        ComponentModel(
            id="comp2",
            type=ComponentType.API_ENDPOINT,
            title="API 1",
            position=Position3D(x=10, y=0, z=0),
            config={"prompt": "test"},
            connections=[],
            svg_style="alert"
        )
    ]
    result = qiskit_engine.build_circuit_from_components(components)
    assert "qasm" in result
    assert "quantum_hash" in result
    assert len(result["quantum_hash"]) == 64
    assert "h q" in result["qasm"] or "x q" in result["qasm"]


def test_run_circuit_with_valid_qasm(qiskit_engine):
    components = [ComponentModel(id="comp1", type=ComponentType.AGENT, title="Agent 1", position=Position3D(x=0, y=0, z=0), config={"vial_id": "vial1"}, connections=[])]
    circuit = qiskit_engine.build_circuit_from_components(components)
    result = qiskit_engine.run_circuit(circuit["qasm"])
    assert "counts" in result
    assert isinstance(result["counts"], dict)
