import math
import numpy as np
from statevectorsim import QuantumState, QuantumCircuit, QuantumGate
from statevectorsim.utils import plot_bloch_spheres, format_statevector

# ==============================================================================
#                      Testing Utility Functions
# ==============================================================================

def run_test_and_plot(title: str, circuit: QuantumCircuit, target_qubits: list[int] = None):
    """
    Initializes a QuantumState (all |0>), runs the circuit, and visualizes
    the result on the Bloch spheres for all qubits in the circuit.
    """
    n_qubits = circuit.n
    print("=" * 70)
    print(f"RUNNING TEST: {title} ({n_qubits} Qubits)")

    # Initialize the state to |0...0>
    state = QuantumState(n_qubits)

    # Run the circuit
    print("Applying Gates...")
    circuit.run(state)

    # Print the resulting statevector for verification
    print(f"Final State Vector: {state.state}")

    # Visualize the Resulting State
    print(f"Plotting Bloch Spheres for {n_qubits} Qubits...")

    # The plot_bloch_spheres function is expected to handle the state vector
    plot_bloch_spheres(state.state)

    print("-" * 70)


# def run_circuit_test(title: str, n_qubits: int, circuit: QuantumCircuit):
#     """
#     Initializes a QuantumState, runs the circuit, and visualizes the result.
#     """
#     print("=" * 70)
#     print(f"RUNNING CIRCUIT TEST: {title} ({n_qubits} Qubits)")
#
#     # Initialize the state to |0...0>
#     state = QuantumState(n_qubits)  # Initializes to |0>^N
#     # Assuming .state is the correct attribute based on the previous fix attempt
#     print(f"Initial State Vector:\n{state.state}")
#
#     # Run the circuit
#     print("\nApplying Gates...")
#     final_state = circuit.run(state)
#
#     # Print Final State Vector
#     print("\n--- Final State Vector (Amplitudes) ---")
#     # Display the state vector using the utility function
#     print(format_statevector(final_state.statevector()))
#
#     # Visualize the Resulting State on Bloch Spheres
#     plot_bloch_spheres(final_state.statevector(), max_cols=n_qubits)

# ==============================================================================
#                      Single-Qubit Gate Tests (1 Qubit)
# ==============================================================================

def test_i_gate():
    """Test Identity (I) gate: |0> -> |0> (no change)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.i(0))
    run_test_and_plot("1. I Gate: |0> -> |0>", qc, target_qubits=[0])

def test_x_gate():
    """Test Pauli-X (NOT) gate: |0> -> |1> (Flips the state)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.x(0))
    run_test_and_plot("2. X Gate: |0> -> |1> (+Z to -Z)", qc, target_qubits=[0])

def test_y_gate():
    """Test Pauli-Y gate: |0> -> i|1> (Rotates by pi about Y-axis to -Z)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.y(0))
    run_test_and_plot("3. Y Gate: |0> -> i|1> (+Z to -Z, with global phase)", qc, target_qubits=[0])

def test_z_gate():
    """Test Pauli-Z gate: |0> -> |0> (No change on |0>, 180 deg phase flip on |1>)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.z(0))
    run_test_and_plot("4. Z Gate: |0> -> |0>", qc, target_qubits=[0])

def test_h_gate():
    """Test Hadamard (H) gate: |0> -> |+> (Superposition, +X axis)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.h(0))
    run_test_and_plot("5. H Gate: |0> -> |+> (+X axis)", qc, target_qubits=[0])

def test_s_gate():
    """Test Phase (S) gate on |+> state: S|+> = (|0> + i|1>)/sqrt(2) (+Y axis)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.h(0)) # |0> -> |+>
    # The S gate (Z rotation by pi/2) rotates |+> from +X to +Y
    qc.add_gate(QuantumGate.s(0))
    run_test_and_plot("6. S Gate on |+> state: +X -> +Y", qc, target_qubits=[0])

def test_t_gate():
    """Test T gate on |+> state: Rotates phase by pi/4 (towards +Y)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.h(0)) # |0> -> |+>
    qc.add_gate(QuantumGate.t(0))
    run_test_and_plot("7. T Gate on |+> state: Phase rotation by pi/4", qc, target_qubits=[0])

def test_rx_pi_2():
    """Test Rx(pi/2) gate: Rotates |0> 90 deg about X-axis to -Y axis."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.rx(0, math.pi / 2))
    run_test_and_plot("8. Rx(pi/2): |0> -> -|y>", qc, target_qubits=[0])

def test_ry_pi_2():
    """Test Ry(pi/2) gate: Rotates |0> 90 deg about Y-axis to +X axis (|0> -> |+>)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.ry(0, math.pi / 2))
    run_test_and_plot("9. Ry(pi/2): |0> -> |+> (+X)", qc, target_qubits=[0])

def test_rz_pi():
    """Test Rz(pi) gate on |+>: Rotates |+> 180 deg about Z-axis to |-> (-X axis)."""
    qc = QuantumCircuit(1)
    qc.add_gate(QuantumGate.h(0)) # |0> -> |+>
    qc.add_gate(QuantumGate.rz(0, math.pi))
    run_test_and_plot("10. Rz(pi) on |+>: +X -> -X", qc, target_qubits=[0])


# ==============================================================================
#                      Two-Qubit Controlled/Swap Gates
# ==============================================================================

def test_bell_state_cx():
    """Test CNOT (CX) gate by creating the Bell state |Φ+>."""
    qc = QuantumCircuit(2)
    qc.add_gate(QuantumGate.h(0))    # |00> -> (|00> + |10>)/sqrt(2)
    qc.add_gate(QuantumGate.cx(0, 1)) # -> (|00> + |11>)/sqrt(2) (Bell state)

    # Entangled state should have both qubits in a mixed state (vector at center)
    run_test_and_plot("11. Bell State (|Φ+>) using H & CX (Q0=C, Q1=T)", qc, target_qubits=[0, 1])

def test_cz_gate():
    """Test CZ gate by acting on the Bell state |Φ+> to get |Ψ->."""
    qc = QuantumCircuit(2)
    qc.add_gate(QuantumGate.h(0))    # H(0)
    qc.add_gate(QuantumGate.cx(0, 1)) # CX(0, 1) -> |Φ+>
    qc.add_gate(QuantumGate.cz(0, 1)) # CZ(|Φ+>) = |Ψ-> = (|00>-|11>)/sqrt(2)

    run_test_and_plot("12. CZ Gate on Bell State (Still entangled)", qc, target_qubits=[0, 1])

def test_swap_gate():
    """Test SWAP gate: Swap states of |01> to |10>."""
    qc = QuantumCircuit(2)
    # Prepare Qubit 1 in |1> state: |00> -> |01>
    qc.add_gate(QuantumGate.x(1))
    # Apply SWAP(0, 1) to get |10>
    qc.add_gate(QuantumGate.swap(0, 1))

    # Qubit 0 should now be |1> (down) and Qubit 1 should be |0> (up)
    run_test_and_plot("13. SWAP Gate: |01> -> |10>", qc, target_qubits=[0, 1])


# ==============================================================================
#                      Controlled Rotation Gates (2 Qubits)
# ==============================================================================

def test_crx_gate():
    """Test CRX(pi) gate: Apply X (180 deg rotation) to target if control is |1>."""
    qc = QuantumCircuit(2)
    qc.add_gate(QuantumGate.x(0))      # Set control Q0 to |1>. State: |10>
    qc.add_gate(QuantumGate.crx(0, 1, math.pi)) # Should apply X to Q1: |10> -> |11>

    # Both Q0 and Q1 should be pointing down (-Z axis)
    run_test_and_plot("14. CRX(pi) on |10>: |10> -> |11>", qc, target_qubits=[0, 1])

def test_cry_gate():
    """Test CRY(pi) gate: Apply Y (180 deg rotation) to target if control is |1>."""
    qc = QuantumCircuit(2)
    qc.add_gate(QuantumGate.x(0))      # Set control Q0 to |1>. State: |10>
    qc.add_gate(QuantumGate.cry(0, 1, math.pi)) # Should apply Y to Q1: |10> -> i|11>

    # Both Q0 and Q1 should be pointing down (-Z axis)
    run_test_and_plot("15. CRY(pi) on |10>: |10> -> i|11>", qc, target_qubits=[0, 1])

def test_crz_gate():
    """Test CRZ(pi) gate: Apply Z (180 deg rotation) to target if control is |1>."""
    qc = QuantumCircuit(2)
    # Prepare a state where RZ is noticeable on Q1: |1+> = (|10> + |11>)/sqrt(2)
    qc.add_gate(QuantumGate.x(0))
    qc.add_gate(QuantumGate.h(1))      # State: |1+>

    # CRZ(pi) should apply Z to Q1: |1+> -> |1-> (Q1 moves from +X to -X)
    qc.add_gate(QuantumGate.crz(0, 1, math.pi))

    # Q0: |1> (-Z); Q1: |-> (-X)
    run_test_and_plot("16. CRZ(pi) on |1+>: Q1 is flipped +X -> -X", qc, target_qubits=[0, 1])


# ==============================================================================
#                      Multi-Controlled Gates (3 Qubits)
# ==============================================================================

def test_mcx_gate():
    """Test Toffoli (CCX) gate: Multi-Controlled X (2 controls, 1 target)."""
    qc = QuantumCircuit(3)
    # Set controls Q0 and Q1 to |1>. State: |110>
    qc.add_gate(QuantumGate.x([0, 1]))
    # Apply MCX(0, 1, target=2): |110> -> |111>
    qc.add_gate(QuantumGate.mcx([0, 1], 2))

    # All qubits should be pointing down (-Z axis)
    run_test_and_plot("17. MCX (Toffoli): |110> -> |111>", qc, target_qubits=[0, 1, 2])

def test_mcy_gate():
    """Test Multi-Controlled Y gate (2 controls, 1 target)."""
    qc = QuantumCircuit(3)
    # Set controls Q0 and Q1 to |1>. State: |110>
    qc.add_gate(QuantumGate.x([0, 1]))
    # Apply MCY(0, 1, target=2): |110> -> i|111>
    qc.add_gate(QuantumGate.mcy([0, 1], 2))

    # All qubits should be pointing down (-Z axis)
    run_test_and_plot("18. MCY: |110> -> i|111>", qc, target_qubits=[0, 1, 2])

def test_mcz_gate():
    """Test Multi-Controlled Z gate (2 controls, 1 target)."""
    qc = QuantumCircuit(3)
    # Prepare state |11+> where RZ is noticeable on Q2:
    qc.add_gate(QuantumGate.x([0, 1]))
    qc.add_gate(QuantumGate.h(2))      # State: |11+>

    # MCZ should apply Z to Q2: |11+> -> -|11-> (Q2 moves from +X to -X)
    qc.add_gate(QuantumGate.mcz([0, 1], 2))

    # Q0, Q1: |1> (-Z); Q2: |-> (-X)
    run_test_and_plot("19. MCZ on |11+>: Q2 is flipped +X -> -X", qc, target_qubits=[0, 1, 2])


# ==============================================================================
#                               Circuit Tests
# ==============================================================================

def test_qft_decomposition(n_qubits: int, initial_index: int):
    """
    Test the QFT implementation by applying it to a basis state |x> and checking against analytical result.
    """

    # Pad the index for printing
    x_str = bin(initial_index)[2:].zfill(n_qubits)
    print(f"--- Testing QFT Decomposition for |{x_str}> ({n_qubits} qubits) ---")

    # Define the input state |x>
    initial_state = QuantumState(n_qubits)
    initial_state.basis_state(initial_index)

    # Build the QFT circuit. QFT swaps endian so reverse.
    qft_circuit = QuantumCircuit.qft(n_qubits, swap_endian=True)

    # Run the circuit
    final_state = qft_circuit.run(initial_state)

    # Define the expected output state (analytical result)
    expected_state = np.zeros(2**n_qubits, dtype=complex)
    N = 2**n_qubits
    x = initial_index

    # Calculate expected amplitudes for each basis state |k>
    for k in range(N):
        # Calculate the phase: 2 * pi * x * k / N
        phase_angle = 2 * np.pi * x * k / N
        # Amplitude is (1/sqrt(N)) * exp(i * phase_angle)
        expected_state[k] = (1 / np.sqrt(N)) * np.exp(1j * phase_angle)

    # Assert: Check if the final state matches the expected state
    TOLERANCE = 1e-7
    assert np.allclose(final_state.state, expected_state, atol=TOLERANCE), (
        f"QFT state mismatch for {n_qubits} qubits on |{x_str}>.\n"
        f"Expected:\n{expected_state}\n"
        f"Got:\n{final_state.state}\n"
    )

    print(f"QFT Test ({n_qubits} qubits on |{x_str}>) PASSED.")
    print("--- QFT Test Results (First 8 amplitudes) ---")
    print(f"Final State Vector: {final_state.state[:8]}...")

    plot_bloch_spheres(final_state.state)


# ==============================================================================
#                              Main Test Suite
# ==============================================================================

def test_all_gates():
    """
    Main function to run all functional and visualization tests.
    """
    print("=" * 70)
    print("   QUANTUM GATE FUNCTIONAL AND BLOCH SPHERE VISUALIZATION TESTS")
    print("=" * 70)

    # Single-Qubit Standard Gates
    test_i_gate()
    test_x_gate()
    test_y_gate()
    test_z_gate()
    test_h_gate()
    test_s_gate()
    test_t_gate()

    # Single-Qubit Rotation Gates
    test_rx_pi_2()
    test_ry_pi_2()
    test_rz_pi()

    # Two-Qubit Standard/Swap Gates
    test_bell_state_cx()
    test_cz_gate()
    test_swap_gate()

    # Two-Qubit Controlled Rotation Gates
    test_crx_gate()
    test_cry_gate()
    test_crz_gate()

    # Multi-Controlled Gates (3 Qubits)
    test_mcx_gate()
    test_mcy_gate()
    test_mcz_gate()

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE: Check console output and Bloch Sphere plots for all gates.")
    print("=" * 70)


if __name__ == "__main__":
    test_all_gates()
    test_qft_decomposition(4, 7)