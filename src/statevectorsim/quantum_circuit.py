import numpy as np

# Assuming these are defined elsewhere
from .quantum_state import QuantumState
from .quantum_gate import QuantumGate


class QuantumCircuit:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.gates: list[QuantumGate] = []

    def reset(self):
        """ Clears all gates from the circuit, returning to empty state."""
        self.gates = []

    def copy(self):
        """ Creates a new deep-copy QuantumCircuit instance """
        new_qc = QuantumCircuit(self.n)
        # Copy the list of references to the existing gate objects
        new_qc.gates = list(self.gates)
        return new_qc

    def add_gate(self, gate):
        """
        Add a single QuantumGate or a list/tuple of QuantumGate objects to circuit.
        e.g. qc.add_gate(QuantumGate.h([0, 1, 2]))
        """

        # list of gates
        if isinstance(gate, list) or isinstance(gate, tuple):
            self.gates.extend(gate)

        # single gate
        else:
            self.gates.append(gate)

    def run(self, quantum_state, inverse=False):
        """ Apply quantum gates to state in order """

        # forward
        if not inverse:
            for gate in self.gates:
                gate.apply(quantum_state)

        # inverse
        else:
            # Apply the inverse sequence of gates (order reversed)
            for gate in reversed(self.gates):
                # We assume the gate itself handles the inverse application if needed
                gate.apply(quantum_state)

        return quantum_state

    def simulate(self, initial_state: QuantumState, shots: int = 1024) -> dict[str, int]:
        """
        Runs the circuit multiple times and returns a dictionary of measurement outcomes.
        """
        if initial_state.n != self.n:
            raise ValueError(f"Initial state must have the same number of qubits as the circuit.")

        results = {}

        for _ in range(shots):
            # create fresh copy of the initial state for each shot
            current_state = initial_state.copy()

            # run circuit
            self.run(current_state)

            # measure all qubits
            # Assuming 'measure_all' is a method on QuantumState that returns a list of bits
            outcome_list = current_state.measure_all()

            # convert the list of bits into a measurement key (e.g., '01')
            outcome_str = "".join(map(str, outcome_list))

            # record the count
            results[outcome_str] = results.get(outcome_str, 0) + 1

        return results

    @staticmethod
    def bell():
        """
        Return 2-qubit Bell state circuit (|Φ+⟩ = (|00⟩ + |11⟩)/√2).
        """
        _qc = QuantumCircuit(2)
        _qc.add_gate(QuantumGate.h(0))  # Hadamard on qubit 0
        _qc.add_gate(QuantumGate.cx(0, 1))  # CNOT control=0, target=1
        return _qc

    @staticmethod
    def ghz(n_qubits=3):
        """
        Return a GHZ state circuit for n_qubits (|GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2). n >= 2.
        """
        if n_qubits < 2:
            raise ValueError("GHZ state requires at least 2 qubits.")

        _qc = QuantumCircuit(n_qubits)
        _qc.add_gate(QuantumGate.h(0))  # Hadamard on the first qubit

        # CNOT chain to entangle the rest
        for i in range(n_qubits - 1):
            _qc.add_gate(QuantumGate.cx(i, i + 1))  # Control i, Target i+1

        return _qc

    @staticmethod
    def qft(n_qubits: int, swap_endian: bool = False, inverse: bool = False):
        """
        Return n-qubit Quantum Fourier Transform (QFT) circuit.
        Implementation is MSB-first.
        """

        if n_qubits < 1:
            raise ValueError("QFT requires at least 1 qubit.")

        _qc = QuantumCircuit(n_qubits)

        if not inverse:
            # --- Forward QFT ---
            # Process from MSB (n-1) down to LSB (0)
            for i in reversed(range(n_qubits)):

                _qc.add_gate(QuantumGate.h(i))

                # Apply controlled rotations from indexed qubits j < i
                for j in reversed(range(i)):

                    theta = 2 * np.pi / (2 ** (i - j + 1))

                    gate = QuantumGate.crp(j, i, theta)

                    _qc.add_gate(gate)

        else:
            # --- Inverse QFT ---
            # Process from LSB (0) up to MSB (n-1)
            for i in range(n_qubits):

                # Apply inverse controlled rotations from lower indexed qubits j < i
                for j in range(i):
                    theta = -2 * np.pi / (2 ** (i - j + 1))
                    gate = QuantumGate.crp(j, i, theta)
                    _qc.add_gate(gate)

                _qc.add_gate(QuantumGate.h(i))

        # swap gates to reverse order
        if swap_endian:
            for i in range(n_qubits // 2):
                _qc.add_gate(QuantumGate.swap(i, n_qubits - 1 - i))

        return _qc

    @staticmethod
    def _prepare_counting_register(qc: 'QuantumCircuit', start_index: int, end_index: int):
        """Applies Hadamard to a range of qubits."""
        for i in range(start_index, end_index):
            qc.add_gate(QuantumGate.h(i))
