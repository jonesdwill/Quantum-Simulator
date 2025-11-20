import numpy as np
import math

class QuantumState:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.basis_state()

    def basis_state(self):
        self.state[:] = 0
        self.state[0] = 1.0

    def statevector(self):
        return self.state.copy()

    def get_probabilities(self):
        return np.abs(self.state) ** 2

    def measure_qubit(self, qubit: int):
        # re-shape statevector into an n-dimensional tensor and compute probabilities
        probability_tensor = np.abs(self.state.reshape([2] * self.n)) ** 2

        # permute axes to be able to slice measured qubit across all states
        permutation = [qubit] + [i for i in range(self.n) if i != qubit]
        permuted = np.transpose(probability_tensor, permutation)

        # sum over all amplitudes where measured qubit = 0 to get probability p0
        p0 = np.sum(permuted[0])

        # randomly sample 0 or 1 for qubit.
        outcome = np.random.choice([0, 1], p=[p0, 1 - p0])

        # create mask for statevector to force state to remain in place after measurement
        mask = np.zeros([2] * self.n)
        mask_tuple = (outcome,) + tuple(slice(None) for _ in range(self.n - 1))
        mask[mask_tuple] = 1

        # undo permutation to match original statevector axes
        mask = np.transpose(mask, np.argsort(permutation))

        # collapse statevector to measured state
        new_state = (self.state.reshape([2] * self.n) * mask).reshape(self.dim)
        self.state = new_state / np.linalg.norm(new_state) # normalise

        return outcome

    def measure_all(self):
        """
        Measure all qubits in the computational basis.
        """

        # re-shape statevector into an n-dimensional tensor and compute probabilities
        probability_tensor = np.abs(self.state.reshape([2] * self.n)) ** 2

        # flatten tensor to get probability for each computational basis state
        probability_vector = probability_tensor.flatten()

        # sample one state according to the probabilities
        index = np.random.choice(len(probability_vector), p=probability_vector)

        # convert index to bitstring
        outcome = [(index >> i) & 1 for i in reversed(range(self.n))]

        # collapse statevector to measured basis state
        new_state = np.zeros_like(self.state)
        new_state[index] = 1.0
        self.state = new_state

        return outcome