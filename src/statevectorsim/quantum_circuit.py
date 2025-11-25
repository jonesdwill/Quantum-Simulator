import numpy as np
import math
from typing import List
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

    # ---------------------------------------------------------
    #           Entanglement Circuits (Bell & GHZ)
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    #           Quantum Fourier Transform (QFT)
    # ---------------------------------------------------------

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

                    theta = 2 * np.pi / (2 **    (i - j + 1))

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


    # ---------------------------------------------------------
    #           Quantum Phase Estimation (QPE)
    # ---------------------------------------------------------

    @staticmethod
    def _prepare_counting_register(qc: 'QuantumCircuit', t_qubits: int):
        """Applies Hadamard to the counting register (qubits 0 to t_qubits - 1)."""
        for i in range(t_qubits):
            qc.add_gate(QuantumGate.h(i))

    @staticmethod
    def qpe(t_qubits: int, unitary_matrix: np.ndarray, m_qubits: int,  target_initial_state_gates: List['QuantumGate'] = None) -> 'QuantumCircuit':
        """
        Return a Quantum Phase Estimation (QPE) circuit.

        Args:
            t_qubits (int): The size of the counting register (qubits 0 to t-1).
            unitary_matrix (np.ndarray): The 2^m x 2^m unitary matrix U.
            m_qubits (int): The size of the target register.
            target_initial_state_gates (List[QuantumGate]):
                    Gates to prepare the target state |psi>, which must be an eigenvector of U.

        Returns:
            QuantumCircuit: The complete QPE circuit.
        """
        n_qubits = t_qubits + m_qubits

        if unitary_matrix.shape != (2 ** m_qubits, 2 ** m_qubits):
            raise ValueError(
                f"Unitary matrix size ({unitary_matrix.shape}) must be 2^m x 2^m where m is "
                f"the number of target qubits ({m_qubits})."
            )

        _qc = QuantumCircuit(n_qubits)

        # Determine indices for target reg
        target_indices = list(range(t_qubits, n_qubits))  # Indices from t to n-1

        # prep target reg
        if target_initial_state_gates:
            for gate in target_initial_state_gates:
                _qc.add_gate(gate)

        # prep counting reg
        QuantumCircuit._prepare_counting_register(_qc, t_qubits)

        # Apply C-U^(2^j) operations. iterate to get 2^0, 2^1, ..., 2^(t-1).
        # The control qubit for the largest power (2^(t-1)) is the LSB of the iteration (qubit 0).
        for j in range(t_qubits):

            _k = 2 ** j

            # j=0 -> control=t-1 (LSB of counting register)
            # j=t-1 -> control=0 (MSB of counting register)
            control_qubit = t_qubits - 1 - j

            # Create and add the Controlled-U^power gate
            gate = QuantumGate.cu(
                control=control_qubit,
                target_qubits=target_indices,
                unitary_matrix=unitary_matrix,
                k=_k  # using 'k' for power as per your cu signature
            )
            _qc.add_gate(gate)

        # Apply inverse qft to the counting register.
        iqft_qc = QuantumCircuit.qft(t_qubits, inverse=True, swap_endian=False)
        _qc.gates.extend(iqft_qc.gates)

        return _qc


    # ---------------------------------------------------------
    #                     Grover's Search
    # ---------------------------------------------------------

    @staticmethod
    def _grover_diffuser(qc: 'QuantumCircuit', qubits: List[int]):
        """
        Applies the n-qubit Grover diffusion operator D.
        D = H^n (2|0...0><0...0| - I) H^n
        This is equivalent to: H^n X^n C^nZ X^n H^n
        Where C^nZ is the Multi-Controlled Z (MCZ) gate on all qubits.
        """
        n_qubits = len(qubits)

        # 1. Hadamard on all qubits (H^n)
        for q in qubits:
            qc.add_gate(QuantumGate.h(q))

        # 2. Invert all qubits (X^n)
        for q in qubits:
            qc.add_gate(QuantumGate.x(q))

        # 3. Multi-Controlled Z (C^nZ)
        # The MCZ gate on n qubits acts as the reflection about the |1...1> state.
        # It flips the phase of the |1...1> state and leaves all others unchanged.
        # Note: We must use the MCZ on the entire register.

        # If n_qubits=1, it's just a Z gate. If n_qubits=2, it's CZ.
        if n_qubits >= 2:
            controls = qubits[:-1]
            target = qubits[-1]
            # Since MCZ uses all qubits as controls and a target,
            # we can use the last qubit as the "target" for the MCZ matrix
            # construction, but it effectively acts on all.
            mc_z_gate = QuantumGate.mcz(controls, target)
            qc.add_gate(mc_z_gate)
        elif n_qubits == 1:
            qc.add_gate(QuantumGate.z(qubits[0]))

        # 4. Invert all qubits again (X^n)
        for q in qubits:
            qc.add_gate(QuantumGate.x(q))

        # 5. Hadamard on all qubits (H^n)
        for q in qubits:
            qc.add_gate(QuantumGate.h(q))

    @staticmethod
    def grover_search(
            n_qubits: int,
            marked_state_index: int
    ) -> 'QuantumCircuit':
        """
        Creates a QuantumCircuit for Grover's Search Algorithm.

        Args:
            n_qubits (int): The number of qubits in the search space (N=2^n).
            marked_state_index (int): The decimal index of the unique state |w>
                                      to be searched (0 to 2^n - 1).

        Returns:
            QuantumCircuit: The complete Grover's circuit.
        """
        if n_qubits < 2:
            raise ValueError("Grover's search is typically used for n_qubits >= 2.")

        _qc = QuantumCircuit(n_qubits)
        qubit_indices = list(range(n_qubits))

        # --- 1. Initialization (Hadamard on all qubits) ---
        for q in qubit_indices:
            _qc.add_gate(QuantumGate.h(q))

        # --- 2. Determine Number of Iterations (R) ---
        N = 2 ** n_qubits
        # Optimal number of iterations: R = round(pi/4 * sqrt(N))
        R = round(math.pi / 4 * math.sqrt(N))
        print(f"Optimal Grover iterations (R): {R}")

        # --- 3. Iterative Amplitude Amplification (R times) ---
        for r in range(R):

            # --- 3a. Phase Oracle (O) ---
            # Creates a phase flip on the marked state |w>.
            # O |x> = -|x> if x = w, and |x> otherwise.

            # The oracle can be built using H, X, and MCZ/MCX gates.
            # We use the X-MCZ-X construction to flip the phase of a non-|1...1> state.

            # Find the binary representation of the marked index
            binary_marked = format(marked_state_index, f'0{n_qubits}b')

            # Apply X gates to wrap the Multi-Controlled Z gate (MCZ)
            # The X gates are applied to qubits that correspond to a '0' in the marked state.
            for q in qubit_indices:
                # Qubit q corresponds to bit n_qubits - 1 - q (MSB for index 0)
                # Note: State is LSB-first (q0 is least significant bit)
                # Bit index is q
                if binary_marked[n_qubits - 1 - q] == '0':
                    _qc.add_gate(QuantumGate.x(q))

            # Apply the Multi-Controlled Z (MCZ) to flip the phase of the target |1..1>
            # The controls are all qubits except the last one, and the last qubit is the target.
            if n_qubits >= 2:
                controls = qubit_indices[:-1]
                target = qubit_indices[-1]
                mc_z_gate = QuantumGate.mcz(controls, target)
                _qc.add_gate(mc_z_gate)
            elif n_qubits == 1:  # Single qubit search space N=2
                _qc.add_gate(QuantumGate.z(0))  # Z acts as Oracle for |1> (if we wrap X's)

            # Apply X gates again to un-wrap the MCZ, targeting the marked state |w>
            for q in qubit_indices:
                if binary_marked[n_qubits - 1 - q] == '0':
                    _qc.add_gate(QuantumGate.x(q))

            # --- 3b. Diffusion Operator (D) ---
            QuantumCircuit._grover_diffuser(_qc, qubit_indices)

        return _qc