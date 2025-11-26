import time
import matplotlib.pyplot as plt
from statevectorsim import QuantumState, QuantumCircuit, QuantumGate
from typing import List, Tuple, Dict
import numpy as np

from qiskit.circuit.library import QFT as qiskit_qft
from qiskit.compiler import transpile as qiskit_transpile
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit_aer import Aer
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library import ZGate


# --- QFT Circuit Generator ---

def generate_qft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Generates the custom QFT circuit (the one currently defined in QuantumCircuit.qft).
    This serves as the benchmark reference circuit.
    """
    # Assuming this factory method exists and builds the circuit correctly
    return QuantumCircuit.qft(n_qubits, swap_endian=True, inverse=False)

def generate_qpe_circuit(n_estimation: int) -> QuantumCircuit:
    """
    Generates the custom QPE circuit using QuantumCircuit.qpe.

    Args:
        n_estimation (int): The size of the counting register (t_qubits).

    Uses U=Z on 1 target qubit (|1> eigenstate, phase 0.5).
    """

    z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    m_qubits = 1

    target_qubit_index = n_estimation
    target_initial_state_gates = [QuantumGate.x(target_qubit_index)]

    # Calls the user's comprehensive QPE static method
    return QuantumCircuit.qpe(
        t_qubits=n_estimation,
        unitary_matrix=z_matrix,
        m_qubits=1,
        target_initial_state_gates=target_initial_state_gates
    )

# --- Benchmarking Logic ---

def benchmark_qft(qubit_range: List[int], shots: int) -> Dict[str, List[Tuple[int, float]]]:
    """
    Benchmarks the QFT circuit for different qubit counts using:
    1. 'tensor' method
    2. 'bitmask' method
    3. 'qiskit_aer' method
    """
    results: Dict[str, List[Tuple[int, float]]] = {
        'tensor': [],
        'bitmask': [],
        'qiskit_aer': []
    }

    # Pre-fetch the Aer backend once
    try:
        aer_backend = Aer.get_backend('statevector_simulator')
        aer_backend_is_ready = True
    except Exception as e:
        print(f"FATAL: Failed to initialize Qiskit Aer backend. Error: {e}")
        aer_backend_is_ready = False

    for n in qubit_range:

        print(f"\n--- Benchmarking QFT on {n} Qubits (State Space: 2^{n}={2 ** n}) ---")

        # build local circuit simulator
        qc = generate_qft_circuit(n)

        # --- Benchmark 'tensor' method ---

        total_time_tensor = 0.0
        for i in range(shots):

            state = QuantumState(n)
            state.basis_state(0)
            state_copy = state.copy()

            start_time = time.time()
            qc.run(state_copy, method='tensor')
            end_time = time.time()
            total_time_tensor += (end_time - start_time)

        avg_time_tensor = total_time_tensor / shots
        results['tensor'].append((n, avg_time_tensor))
        print(f"Tensor Only (Forced): Avg time over {shots} shots: {avg_time_tensor:.6f} s")

        # --- Benchmark 'bitmask' method ---
        total_time_bitmask = 0.0
        for i in range(shots):

            state = QuantumState(n)
            state.basis_state(0)
            state_copy = state.copy()

            start_time = time.time()
            qc.run(state_copy, method='bitmask')
            end_time = time.time()
            total_time_bitmask += (end_time - start_time)

        avg_time_bitmask = total_time_bitmask / shots
        results['bitmask'].append((n, avg_time_bitmask))
        print(f"Bitmask Only (Forced): Avg time over {shots} shots: {avg_time_bitmask:.6f} s")

        # --- Benchmark 'Qiskit Aer' method ---
        if not aer_backend_is_ready:
            continue

        total_time_aer = 0.0

        try:
            # Create Qiskit Circuit using the qiskit library
            qiskit_qft_qc = QiskitCircuit(n)

            # Append QFT object directly
            qiskit_qft_qc.append(qiskit_qft(n, do_swaps=True), range(n))

            # Transpile to Aer's basis gates to ensure it's runnable
            transpiled_qc = qiskit_transpile(qiskit_qft_qc, aer_backend)

            for i in range(shots):
                start_time = time.time()

                # Execute transpiled circuit
                job = aer_backend.run(transpiled_qc, shots=1)
                result = job.result()

                # Ensure execution is complete
                _ = result.get_statevector()

                end_time = time.time()
                total_time_aer += (end_time - start_time)

            avg_time_aer = total_time_aer / shots
            results['qiskit_aer'].append((n, avg_time_aer))
            print(f"Qiskit Aer Statevector: Avg time over {shots} shots: {avg_time_aer:.6f} s")

        except Exception as e:
            print(f"WARNING: Qiskit Aer benchmark failed for N={n}. Error: {e}")

    return results

def benchmark_qpe(qubit_range: List[int], shots: int) -> Dict[str, List[Tuple[int, float]]]:
    """
    Benchmarks the QFT circuit for different qubit counts using:
    1. 'tensor' method
    2. 'bitmask' method
    3. 'qiskit_aer' method
    """
    results: Dict[str, List[Tuple[int, float]]] = {
        'tensor': [],
        'bitmask': [],
        'qiskit_aer': []
    }

    # Pre-fetch the Aer backend once
    try:
        aer_backend = Aer.get_backend('statevector_simulator')
        aer_backend_is_ready = True
    except Exception as e:
        print(f"FATAL: Failed to initialize Qiskit Aer backend. Error: {e}")
        aer_backend_is_ready = False

    for n in qubit_range:
        n_total = n + 1

        print(f"\n--- Benchmarking QPE on {n} Estimation Qubits ({n_total} Total Qubits) ---")

        # build local circuit simulator
        qc = generate_qpe_circuit(n)

        # --- Benchmark 'tensor' method ---

        total_time_tensor = 0.0
        for i in range(shots):

            state = QuantumState(n_total)
            state.basis_state(0)
            state_copy = state.copy()

            start_time = time.time()
            qc.run(state_copy, method='tensor')
            end_time = time.time()
            total_time_tensor += (end_time - start_time)

        avg_time_tensor = total_time_tensor / shots
        results['tensor'].append((n, avg_time_tensor))
        print(f"Tensor Only (Forced): Avg time over {shots} shots: {avg_time_tensor:.6f} s")

        # --- Benchmark 'bitmask' method ---
        total_time_bitmask = 0.0
        for i in range(shots):

            state = QuantumState(n_total)
            state.basis_state(0)
            state_copy = state.copy()

            start_time = time.time()
            qc.run(state_copy, method='bitmask')
            end_time = time.time()
            total_time_bitmask += (end_time - start_time)

        avg_time_bitmask = total_time_bitmask / shots
        results['bitmask'].append((n, avg_time_bitmask))
        print(f"Bitmask Only (Forced): Avg time over {shots} shots: {avg_time_bitmask:.6f} s")

        # --- Benchmark 'Qiskit Aer' method ---
        if not aer_backend_is_ready:
            continue

        total_time_aer = 0.0

        try:
            # Create Qiskit Circuit using the qiskit library
            qiskit_qpe_qc = QiskitCircuit(n_total)

            # Append QPE object directly
            qiskit_qpe_qc.append(PhaseEstimation(n, ZGate()), range(n_total))

            # Transpile to Aer's basis gates to ensure it's runnable
            transpiled_qc = qiskit_transpile(qiskit_qpe_qc, aer_backend)

            for i in range(shots):
                start_time = time.time()

                # Execute transpiled circuit
                job = aer_backend.run(transpiled_qc, shots=1)
                result = job.result()

                # Ensure execution is complete
                _ = result.get_statevector()

                end_time = time.time()
                total_time_aer += (end_time - start_time)

            avg_time_aer = total_time_aer / shots
            results['qiskit_aer'].append((n, avg_time_aer))
            print(f"Qiskit Aer Statevector: Avg time over {shots} shots: {avg_time_aer:.6f} s")

        except Exception as e:
            print(f"WARNING: Qiskit Aer benchmark failed for N={n}. Error: {e}")

    return results

def plot_benchmarks(results: Dict[str, List[Tuple[int, float]]], shots: int, max_qubits: int, circuit_title = 'QFT'):
    """
    Generates a matplotlib plot of simulation time vs. number of qubits.
    """
    plt.figure(figsize=(10, 6))

    # Define custom labels for the legend
    labels = {
        'tensor': 'Tensor Only',
        'bitmask': 'Bitmask Only',
        'qiskit_aer': r"IBM's Qiskit Aer Statevector Simulator"
    }

    for method, data in results.items():
        if not data:
            continue

        n_qubits = [d[0] for d in data]
        times = [d[1] for d in data]
        plt.plot(n_qubits, times, marker='o', linestyle='-', label=labels.get(method, method))

    plt.title(f'{circuit_title} Benchmark Comparison: {max_qubits} Qubits, Avg over {shots} Shots')
    plt.xlabel('Number of Qubits (n)')
    plt.ylabel('Average Simulation Time (seconds) - Log Scale')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ensure all tested qubit counts are on the x-axis, using a list comprehension for safety
    all_n = sorted(list(set([d[0] for d in results.get('tensor', [])] + [d[0] for d in results.get('qiskit_aer', [])])))
    if all_n:
        plt.xticks(all_n)

    plt.yscale('log')  # Use log scale to better visualize exponential growth
    plt.tight_layout()
    plt.savefig(f'{circuit_title}_benchmark.png')


# --- Main Execution Setup ---
if __name__ == '__main__':
    # Define the range of qubits to test.
    QUBIT_RANGE = list(range(2, 8))  # 2 to 7 qubits (2^7 = 128 dimensions max)
    NUM_SHOTS = 10

    print("Starting QFT Benchmarking (including Qiskit-based Hybrid Method)...")

    # Run the benchmark
    benchmark_results = benchmark_qft(QUBIT_RANGE, NUM_SHOTS)

    # Plot the results
    plot_benchmarks(benchmark_results, NUM_SHOTS, QUBIT_RANGE[-1])

    # Define the range of qubits to test.
    MAX_NUM_QUBITS = 7
    # Define the range of qubits to test.
    QUBIT_RANGE = list(range(1, MAX_NUM_QUBITS + 1))  # 2 to 7 qubits (2^7 = 128 dimensions max)
    NUM_SHOTS = 100

    print("Starting QPE Benchmarking (including Qiskit-based Hybrid Method)...")

    # Run the benchmark
    benchmark_results = benchmark_qpe(QUBIT_RANGE, NUM_SHOTS)