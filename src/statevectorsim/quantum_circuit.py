from .quantum_state import QuantumState
from .quantum_gate import QuantumGate

class QuantumCircuit:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.gates: list[QuantumGate] = []

    def add_gate(self, gate: QuantumGate):
        self.gates.append(gate)

    def run(self, quantum_state: QuantumState):
        for gate in self.gates:
            gate.apply(quantum_state)