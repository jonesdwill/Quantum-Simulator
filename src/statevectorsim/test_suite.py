import numpy as np
from statevectorsim import QuantumState, QuantumCircuit, QuantumGate
from statevectorsim.utils import *

def assert_gate(gate, expected_targets, expected_dim, expected_matrix=None):
    """Utility function for common assertions on a QuantumGate object."""
    assert isinstance(gate, QuantumGate), f"Result is not a QuantumGate object: {gate}"
    assert gate.targets == expected_targets, f"Targets mismatch: Expected {expected_targets}, got {gate.targets}"
    assert gate.matrix.shape == (expected_dim,
                                 expected_dim), f"Matrix shape mismatch: Expected ({expected_dim}, {expected_dim}), got {gate.matrix.shape}"

    if expected_matrix is not None:
        # Check if the matrix content is correct
        assert np.allclose(gate.matrix, expected_matrix), f"Matrix content mismatch for targets {expected_targets}."


def test_single_gates():
    print("--- Testing Single Qubit Gates (X, Y, Z, H, I) ---")

    # Matrices for comparison
    I_mat = np.eye(2, dtype=complex)
    X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
    H_mat = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

    # Test single qubit
    assert_gate(QuantumGate.i(5), [5], 2, I_mat)
    assert_gate(QuantumGate.x(0)[0], [0], 2, X_mat)
    assert_gate(QuantumGate.y(1)[0], [1], 2, Y_mat)
    assert_gate(QuantumGate.z(2)[0], [2], 2, Z_mat)
    assert_gate(QuantumGate.h(3)[0], [3], 2, H_mat)

    # Test multiple qubits (returns list of gates)
    x_gates = QuantumGate.x([4, 5, 6])
    assert len(x_gates) == 3, "Multiple gate call failed to return the correct count."
    assert x_gates[1].targets == [5], "Multiple gate call failed to set correct target."
    assert np.allclose(x_gates[2].matrix, X_mat), "Multiple gate call failed to set correct matrix."

    print("Single Qubit Gates: PASS")


def test_rotation_gates():
    print("--- Testing Single Qubit Rotation Gates (Rx, Ry, Rz) ---")

    theta = math.pi / 2  # 90 degrees rotation

    # R_x(pi/2) matrix
    Rx_mat = np.array([[1 / np.sqrt(2), -1j / np.sqrt(2)], [-1j / np.sqrt(2), 1 / np.sqrt(2)]], dtype=complex)
    # R_y(pi/2) matrix
    Ry_mat = np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]], dtype=complex)
    # R_z(pi/2) matrix
    Rz_mat = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)

    # Test single application
    assert_gate(QuantumGate.rx(0, theta)[0], [0], 2, Rx_mat)
    assert_gate(QuantumGate.ry(1, theta)[0], [1], 2, Ry_mat)
    assert_gate(QuantumGate.rz(2, theta)[0], [2], 2, Rz_mat)

    # Test multiple application
    ry_gates = QuantumGate.ry([3, 4], theta)
    assert len(ry_gates) == 2
    assert_gate(ry_gates[0], [3], 2, Ry_mat)

    print("Rotation Gates: PASS")


def test_two_qubit_gates():
    print("--- Testing Two Qubit Gates (CX, CZ, SWAP) ---")

    # CX Matrix (targets [control, target])
    CX_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    assert_gate(QuantumGate.cx(0, 1), [0, 1], 4, CX_mat)

    # CZ Matrix (targets [control, target])
    CZ_mat = np.diag([1, 1, 1, -1]).astype(complex)
    assert_gate(QuantumGate.cz(2, 3), [2, 3], 4, CZ_mat)

    # SWAP Matrix (targets [q1, q2]) - swaps |01> and |10>
    SWAP_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    assert_gate(QuantumGate.swap(4, 5), [4, 5], 4, SWAP_mat)
    # Check swapped targets: SWAP(5, 4) should yield the same matrix but different target list
    assert_gate(QuantumGate.swap(5, 4), [5, 4], 4, SWAP_mat)

    print("Two Qubit Gates: PASS")


def test_controlled_rotation_gates():
    print("--- Testing Controlled Rotation Gates (CRx, CRy, CRz) ---")
    theta = math.pi  # 180 degrees rotation

    # Rx(pi) = -iX
    Rx_pi_mat = np.array([[0, -1j], [-1j, 0]], dtype=complex)
    # CRx Matrix (targets [c, t]): I block + Rx(pi) block
    CRx_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, 0, -1j], [0, 0, -1j, 0]], dtype=complex)
    assert_gate(QuantumGate.crx(0, 1, theta), [0, 1], 4, CRx_mat)

    # Ry(pi) = -iY
    Ry_pi_mat = np.array([[0, -1], [1, 0]], dtype=complex)
    # CRy Matrix (targets [c, t]): I block + Ry(pi) block
    CRy_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, 0, -1], [0, 0, 1, 0]], dtype=complex)
    assert_gate(QuantumGate.cry(2, 3, theta), [2, 3], 4, CRy_mat)

    # Rz(pi) = -iZ
    Rz_pi_mat = np.diag([-1j, 1j]).astype(complex)  # Rz(pi)
    # CRz Matrix (targets [c, t]): I block + Rz(pi) block
    CRz_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                        [0, 0, -1j, 0], [0, 0, 0, 1j]], dtype=complex)
    assert_gate(QuantumGate.crz(4, 5, theta), [4, 5], 4, CRz_mat)

    print("Controlled Rotation Gates: PASS")


def test_multi_controlled_gates():
    print("--- Testing Multi-Controlled Gates (CCX, MCX, MCY, MCZ) ---")

    # 1. CCX (3-qubit, targets [c1, c2, t])
    CCX_mat = np.eye(8, dtype=complex)
    CCX_mat[6:8, 6:8] = [[0, 1], [1, 0]]
    assert_gate(QuantumGate.ccx(0, 1, 2), [0, 1, 2], 8, CCX_mat)

    # 2. MCX (4-qubit, targets [c1, c2, c3, t])
    mcx_gate = QuantumGate.mcx([0, 1, 2], 3)
    MCX_mat = np.eye(16, dtype=complex)
    MCX_mat[14:16, 14:16] = [[0, 1], [1, 0]]  # Swaps |1110> (14) and |1111> (15)
    assert_gate(mcx_gate, [0, 1, 2, 3], 16, MCX_mat)

    # 3. MCY (4-qubit, targets [c1, c2, c3, t])
    mcy_gate = QuantumGate.mcy([4, 5, 6], 7)
    MCY_mat = np.eye(16, dtype=complex)
    MCY_mat[14:16, 14:16] = [[0, -1j], [1j, 0]]  # Applies Y to |1110> and |1111> block
    assert_gate(mcy_gate, [4, 5, 6, 7], 16, MCY_mat)

    # 4. MCZ (3-qubit, targets [c1, c2, t])
    mcz_gate = QuantumGate.mcz([8, 9], 10)
    MCZ_mat = np.eye(8, dtype=complex)
    MCZ_mat[7, 7] = -1  # Phase flip on |111> (index 7)
    assert_gate(mcz_gate, [8, 9, 10], 8, MCZ_mat)

    print("Multi-Controlled Gates: PASS")


def run_circuit_test(title: str, n_qubits: int, circuit: QuantumCircuit):
    """
    Initializes a QuantumState, runs the circuit, and visualizes the result.
    """
    print("=" * 70)
    print(f"RUNNING CIRCUIT TEST: {title} ({n_qubits} Qubits)")

    # 1. Initialize the state to |0...0>
    state = QuantumState(n_qubits)  # Initializes to |0>^N
    # Assuming .state is the correct attribute based on the previous fix attempt
    print(f"Initial State Vector:\n{state.state}")

    # 2. Run the circuit
    print("\nApplying Gates...")
    final_state = circuit.run(state)

    # Check again just in case the method behavior changes depending on the circuit
    if final_state is None:
        print("\nFATAL ERROR: QuantumCircuit.run() returned None.")
        return

    # 3. Print Final State Vector (FIX: Use print() directly instead of .to_string())
    print("\n--- Final State Vector (Amplitudes) ---")
    print(final_state)  # This calls the object's __str__ method.

    # 4. Visualize the Resulting State on Bloch Spheres
    plot_bloch_spheres(final_state.state)


def example_bell_state():
    """Circuit to create the maximally entangled Bell state |Phi+> = 1/sqrt(2)(|00> + |11>)."""
    n_qubits = 2
    qc = QuantumCircuit(n_qubits)

    # 1. Apply Hadamard to Qubit 1 (creates superposition)
    qc.add_gate(QuantumGate.h(1))

    # 2. Apply CNOT with Control=1, Target=0 (creates entanglement)
    qc.add_gate(QuantumGate.cx(1, 0))  # Targets [control, target]

    run_circuit_test(
        title="Bell State |Phi+> (Entangled)",
        n_qubits=n_qubits,
        circuit=qc
    )


def example_superposition_and_rotation():
    """Circuit to show a qubit moved to the equator and rotated around Z."""
    n_qubits = 1
    qc = QuantumCircuit(n_qubits)

    # 1. Apply Hadamard: |0> -> |+> (Vector points to +X)
    qc.add_gate(QuantumGate.h(0))

    # 2. Apply Rz(pi/2): Rotates the state by 90 degrees around the Z-axis.
    # The state should move from the +X axis towards the +Y axis.
    qc.add_gate(QuantumGate.rz(0, math.pi / 2))

    run_circuit_test(
        title="Superposition -> Rz Rotation (Qubit 0: +X to +Y)",
        n_qubits=n_qubits,
        circuit=qc
    )


def example_mixed_separable():
    """Circuit to show a 3-qubit state where Q0 is |1>, Q1 is |->, and Q2 is |+>."""
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)

    # Qubit 0: |0> -> X -> |1> (Vector points to -Z)
    qc.add_gate(QuantumGate.x(0))

    # Qubit 1: |0> -> H -> |+> -> Z -> |-> (Vector points to -X)
    qc.add_gate(QuantumGate.h(1))
    qc.add_gate(QuantumGate.z(1))

    # Qubit 2: |0> -> H -> |+> (Vector points to +X)
    qc.add_gate(QuantumGate.h(2))

    run_circuit_test(
        title="3-Qubit Separable State (|1>|->|+>)",
        n_qubits=n_qubits,
        circuit=qc
    )

def main():
    try:
        test_single_gates()
        test_rotation_gates()
        test_two_qubit_gates()
        test_controlled_rotation_gates()
        test_multi_controlled_gates()

        print("\nAll Quantum Gate tests passed successfully!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")

    # Run the visualization tests
    example_superposition_and_rotation()
    example_bell_state()
    example_mixed_separable()
