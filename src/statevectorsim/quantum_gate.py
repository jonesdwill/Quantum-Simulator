import numpy as np
from .quantum_state import QuantumState

class QuantumGate:
    """Represents a quantum gate."""
    def __init__(self, matrix: np.ndarray, targets: list[int]):
        self.matrix = matrix
        self.targets = targets  # list of qubits it acts on

    @staticmethod
    def x(target: int):
        """Pauli-X (NOT) gate"""
        return QuantumGate(np.array([[0, 1], [1, 0]], dtype=complex), [target])

    @staticmethod
    def y(target: int):
        """Pauli-Y gate"""
        return QuantumGate(np.array([[0, -1j], [1j, 0]], dtype=complex), [target])

    @staticmethod
    def z(target: int):
        """Pauli-Z gate"""
        return QuantumGate(np.array([[1, 0], [0, -1]], dtype=complex), [target])

    @staticmethod
    def h(target: int):
        """Hadamard gate"""
        return QuantumGate((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex), [target])

    @staticmethod
    def cx(control: int, target: int):
        """CNOT gate (2 qubits)"""
        return QuantumGate(np.array([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,0,1],
                                     [0,0,1,0]], dtype=complex),
                           [control, target])

    @staticmethod
    def i(target: int):
        """Identity gate"""
        return QuantumGate(np.eye(2, dtype=complex), [target])

    def apply(self, quantum_state: QuantumState):
        """Apply gate to QuantumState."""
        n = quantum_state.n
        state_tensor = quantum_state.state.reshape([2] * n)

        # permute target qubits to front
        axes = self.targets + [i for i in range(n) if i not in self.targets]
        permuted = np.transpose(state_tensor, axes)
        permuted = permuted.reshape(2 ** len(self.targets), -1)
        applied = self.matrix @ permuted
        applied = applied.reshape([2] * len(self.targets) + [2] * (n - len(self.targets)))

        # invert permutation
        inv_axes = np.argsort(axes)
        quantum_state.state = np.transpose(applied, inv_axes).reshape(quantum_state.dim)