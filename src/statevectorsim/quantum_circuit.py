import numpy as np
import math
from typing import List
from .quantum_state import QuantumState
from .quantum_gate import QuantumGate
from typing import Union


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

    def add_gate(self, gate: Union[QuantumGate, 'QuantumCircuit', List[QuantumGate]]):
        """
        Add a single QuantumGate, a list/tuple of QuantumGate objects, or
        all gates from another QuantumCircuit instance to the current circuit.
        """

        # QuantumCircuit instance
        if isinstance(gate, QuantumCircuit):
            self.gates.extend(gate.gates)

        # list of gates
        elif isinstance(gate, list) or isinstance(gate, tuple):
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
    def prepare_counting_register(qc: 'QuantumCircuit', t_qubits: int):
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
            QuantumCircuit: complete QPE circuit.
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
        QuantumCircuit.prepare_counting_register(_qc, t_qubits)

        # Apply controlled U^(2^j) operations. iterate 2^0, 2^1, ..., 2^(t-1).
        # NB. The control qubit for the largest power (2^(t-1)) is the LSB of the iteration (qubit 0).
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
    def grover_diffuser(_qc: 'QuantumCircuit', n_qubits: int):
        """
        Applies the n-qubit Grover diffusion operator D.
        D = H^n (2|0...0><0...0| - I) H^n. Equivalent to: H X MCZ X H
        """
        qubits = list(range(n_qubits))

        # Hadamard all qubits to change basis
        for q in qubits:
            _qc.add_gate(QuantumGate.h(q))

        # Invert all qubits
        for q in qubits:
            _qc.add_gate(QuantumGate.x(q))

        # Multi-Controlled Z (C^nZ)
        if n_qubits >= 2: # MCZ for more than one qubit
            controls = qubits[:-1]
            target = qubits[-1] # use last qubit as 'target', though this is just so there is something to direct gate at
            mcz_gate = QuantumGate.mcz(controls, target)
            _qc.add_gate(mcz_gate)

        elif n_qubits == 1: # don't bother with MCZ if just one qubit
            _qc.add_gate(QuantumGate.z(qubits[0]))

        # reverse x gates
        for q in qubits:
            _qc.add_gate(QuantumGate.x(q))

        # reverse basis
        for q in qubits:
            _qc.add_gate(QuantumGate.h(q))

    @staticmethod
    def grover_oracle(_qc: 'QuantumCircuit', n_qubits: int, marked_state_index: int):
        """
        Applies the n-qubit Grover oracle operator O.
        """
        qubits = list(range(n_qubits))

        # binary rep of the marked index
        binary_marked = format(marked_state_index, f'0{n_qubits}b')

        # X gates applied to '0' qubits in the marked state (marked state acts as control '11..11').
        for q in qubits:
            if binary_marked[n_qubits - 1 - q] == '0':
                _qc.add_gate(QuantumGate.x(q))

        # Apply MCZ to flip phase of the target, now |1..1>
        if n_qubits >= 2:
            controls = qubits[:-1]
            target = qubits[-1]
            mc_z_gate = QuantumGate.mcz(controls, target)
            _qc.add_gate(mc_z_gate)
        elif n_qubits == 1:
            _qc.add_gate(QuantumGate.z(0))

            # Apply X gates again to un-wrap MCZ
        for q in qubits:
            if binary_marked[n_qubits - 1 - q] == '0':
                _qc.add_gate(QuantumGate.x(q))

    @staticmethod
    def grover_search(n_qubits: int, marked_state_index: int) -> 'QuantumCircuit':
        """
        Creates QuantumCircuit for Grover's Search.

        Args:
            n_qubits (int): The number of qubits in the search space (N=2^n).
            marked_state_index (int): decimal index of the unique search state |w>
        Returns:
            QuantumCircuit: The complete Grover's circuit.
        """

        if n_qubits < 2:
            raise ValueError("Grover's search is typically used for n_qubits >= 2.")

        _qc = QuantumCircuit(n_qubits)

        # change basis
        for q in range(n_qubits):
            _qc.add_gate(QuantumGate.h(q))

        # determine num iterations. Optimal is R = round(pi/4 * sqrt(N))
        N = 2 ** n_qubits
        R = round(math.pi / 4 * math.sqrt(N))
        print(f"Optimal Grover iterations (R): {R}")

        # Iterate Amplification
        for r in range(R):

            # Phase Oracle (O)
            QuantumCircuit.grover_oracle(_qc, n_qubits, marked_state_index)

            # Diffusion Operator (D)
            QuantumCircuit.grover_diffuser(_qc, n_qubits)

        return _qc

    # ---------------------------------------------------------
    #                     Quantum Adder
    # ---------------------------------------------------------


    @staticmethod
    def qft_adder(n_qubits: int) -> 'QuantumCircuit':
        """
        QFT-based adder. Perform |B>|A> → |B + A mod 2^n>|A>.
          - A : qubits 0 .. n-1   (LSB..MSB)
          - B : qubits n .. 2n-1  (LSB..MSB)
        """
        if n_qubits < 1:
            raise ValueError("n must be >= 1")

        total = 2 * n_qubits
        qc = QuantumCircuit(total)

        # Indices
        A = list(range(0, n_qubits))  # A: 0..n-1 (LSB..MSB)
        B = list(range(n_qubits, 2 * n_qubits))  # B: n..2n-1 (LSB..MSB)

        # Helper: shift gates in a subcircuit by an index offset
        def append_shifted(sub_qc: 'QuantumCircuit', offset: int, target_qc: 'QuantumCircuit'):
            for g in sub_qc.gates:
                new_targets = [t + offset for t in g.targets]
                new_matrix = np.array(g.matrix, copy=True)
                target_qc.add_gate(QuantumGate(new_matrix, new_targets, g.name))

        # --- QFT on B. Shift its targets by +n ---
        qft_on_n = QuantumCircuit.qft(n_qubits, swap_endian=False, inverse=False)
        append_shifted(qft_on_n, offset=n_qubits, target_qc=qc)

        # --- CRP each A & B ---
        for i in range(n_qubits):  # 0 = LSB
            control = A[i]
            for j in range(i, n_qubits):  # only j >= i for non-duplicity
                target = B[j]
                k = j - i + 1
                theta = 2 * np.pi / (2 ** k)
                qc.add_gate(QuantumGate.crp(control, target, theta))

        # --- inverse QFT on B. shift back -n ---
        iqft_on_n = QuantumCircuit.qft(n_qubits, swap_endian=False, inverse=True)
        append_shifted(iqft_on_n, offset=n_qubits, target_qc=qc)

        return qc