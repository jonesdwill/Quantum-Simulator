import numpy as np
from typing import Dict, Union, List, Optional
from .quantum_circuit import QuantumCircuit
from .quantum_state import QuantumState
from .quantum_noise import NoiseModel


class QuantumBackend:
    """
    Intelligent dispatcher that separates compilation (optimization) from execution.
    """

    def __init__(self):
        # Heuristic thresholds
        self.DENSE_QUBIT_LIMIT = 14  # Below this, Dense is almost always faster
        self.SPARSE_CROSSOVER = 20  # Above this, Dense usually runs out of RAM

        # Gates that destroy sparsity rapidly
        self.SCRAMBLING_GATES = {'h', 'rx', 'ry', 'qft'}

        # Default mode, will be overwritten by analysis
        self.mode = 'tbd'

    def analyze_mode(self, circuit: QuantumCircuit) -> str:
        """
        Determines best simulation mode (dense/sparse) and sets self.mode.
        Public method to allow pre-analysis before execution loops.
        """
        n = circuit.n

        # Small circuits -> Dense
        if n <= self.DENSE_QUBIT_LIMIT:
            self.mode = 'dense'
            return 'dense'

        # Large circuits -> Sparse (to avoid MemoryError)
        if n >= self.SPARSE_CROSSOVER:
            self.mode = 'sparse'
            return 'sparse'

        #Medium circuits -> Check structure
        scramble_score = 0
        total_gates = len(circuit.gates)

        for gate in circuit.gates:
            name = gate.name.lower().split('(')[0]
            if name in self.SCRAMBLING_GATES:
                scramble_score += 1

        if total_gates > 0 and (scramble_score / total_gates) > 0.2:
            self.mode = 'dense'
            return 'dense'

        self.mode = 'sparse'
        return 'sparse'

    def optimise_circuit(self, circuit: QuantumCircuit, noise_model: Optional[NoiseModel] = None) -> QuantumCircuit:
        """
        Compiles the circuit for execution.
        """
        # Determine mode for this circuit if not already set
        if self.mode == 'tbd':
            self.analyze_mode(circuit)

        # Work on a copy
        optimised_qc = circuit.copy()

        # Apply Noise
        if noise_model is not None:
            optimised_qc = noise_model.apply(optimised_qc)

        # Optimise (Fusion + Reordering)
        if hasattr(optimised_qc, 'optimise'):
            optimised_qc.optimise()
        else:
            optimised_qc.optimise()

        return optimised_qc

    def execute(self, circuit: QuantumCircuit, initial_state: QuantumState = None, shots: int = 1,inplace: bool = False) -> Union[QuantumState, Dict[str, int]]:
        """
        Runs the circuit exactly as provided (NO optimization).

        Args:
            inplace (bool): If True, modifies initial_state directly. Unsafe for general use.
        """
        # Ensure mode is set
        if self.mode == 'tbd':
            self.analyze_mode(circuit)

        current_mode = self.mode

        # Prepare State
        if initial_state is None:
            state = QuantumState(circuit.n, mode=current_mode)
        else:
            if inplace:
                state = initial_state
            else:
                state = initial_state.copy()

            if state.mode != current_mode:
                if current_mode == 'sparse':
                    state.to_sparse()
                else:
                    state.to_dense()

        # Execute
        if shots == 1:
            circuit.run(state, method=current_mode)
            return state
        else:
            return circuit.simulate(state, shots=shots, method=current_mode)

    # Legacy wrapper for backward compatibility if needed.
    def run(self, circuit: QuantumCircuit, initial_state: QuantumState = None, shots: int = 1, optimise: bool = True,
            noise_model: Optional[NoiseModel] = None) -> Union[QuantumState, Dict[str, int]]:
        """
        One-shot execution wrapper. Optimises and runs.
        """
        qc = circuit
        self.mode = 'tbd'

        if optimise or noise_model:
            qc = self.optimise_circuit(circuit, noise_model)

        return self.execute(qc, initial_state, shots=shots)