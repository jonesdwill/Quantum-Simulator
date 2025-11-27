import matplotlib.pyplot as plt
import time  # Ensure time is imported

from statevectorsim import QuantumState, QuantumCircuit, QuantumGate
from statevectorsim.quantum_backend import QuantumBackend
from typing import List, Tuple, Dict
import numpy as np
import math

# Qiskit Imports
from qiskit.circuit.library import QFT as qiskit_qft
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import transpile  # Added for optimization
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library import ZGate


def plot_benchmarks(results: Dict[str, List[Tuple[int, float]]], shots: int, circuit_title: str):
    """
    Generates a matplotlib plot of simulation time vs. number of qubits.
    """

    all_n = []
    for data in results.values():
        all_n.extend([d[0] for d in data])
    max_qubits = max(all_n) if all_n else 0

    plt.figure(figsize=(10, 6))

    # Extended labels and colors for optimization variants + Hybrid + Qiskit Opt
    labels = {
        'dense': 'Dense',
        'dense_opt': 'Dense (Optimised)',
        'sparse': 'Sparse',
        'sparse_opt': 'Sparse (Optimised)',
        'hybrid': 'Hybrid (Auto-Select)',
        'qiskit_aer': r"IBM's Qiskit Statevector",
        'qiskit_opt': r"IBM's Qiskit Statevector (L3 Opt)"
    }

    styles = {
        'dense': 'o-',
        'dense_opt': 's--',
        'sparse': 'x-',
        'sparse_opt': '^--',
        'hybrid': '*-',
        'qiskit_aer': 'd-',
        'qiskit_opt': 'd--'
    }

    colours = {
        'dense': 'red',
        'dense_opt': 'red',
        'sparse': 'green',
        'sparse_opt': 'green',
        'hybrid': 'blue',
        'qiskit_aer': 'purple',
        'qiskit_opt': 'purple'
    }

    for method, data in results.items():
        if not data:
            continue

        n_qubits = [d[0] for d in data]
        times = [d[1] for d in data]

        style = styles.get(method, method)
        label = labels.get(method, method)
        colour = colours.get(method, method)

        plt.plot(n_qubits, times, style, label=label, color=colour)

    plt.title(f'{circuit_title} Benchmark Comparison: Up to {max_qubits} Qubits, Avg over {shots} Shots')
    plt.xlabel('Number of Qubits (n)')
    plt.ylabel('Average Simulation Time (seconds) - Log Scale')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if all_n:
        plt.xticks(sorted(list(set(all_n))))

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{circuit_title}_benchmark.png')


# --- Benchmarking Logic ----

# ======================================================
#                          QFT
# ======================================================

def generate_qft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Generates custom QFT circuit (QuantumCircuit.qft).
    """
    return QuantumCircuit.qft(n_qubits, swap_endian=True, inverse=False)


def benchmark_qft(qubit_range: List[int], shots: int, methods_to_run: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Benchmarks the QFT circuit. Supports 'hybrid', '_opt' suffix, and 'qiskit_opt'.
    """
    results: Dict[str, List[Tuple[int, float]]] = {}
    for method in methods_to_run:
        results[method] = []

    # Initialize hybrid backend once (if needed)
    hybrid_backend = QuantumBackend()

    for n in qubit_range:

        print(f"\n--- Benchmarking QFT on {n} Qubits (State Space: 2^{n}={2 ** n}) ---")

        # build base local circuit simulator
        base_qc = generate_qft_circuit(n)

        # --- Iterative Benchmarking ---
        for method in methods_to_run:

            # --- Benchmark 'Qiskit Statevector' (Standard) ---
            if method == 'qiskit_aer':
                total_time_aer = 0.0
                try:
                    qiskit_qft_qc = QiskitCircuit(n)
                    qiskit_qft_qc.append(qiskit_qft(n, do_swaps=True), range(n))

                    for i in range(shots):
                        start_time = time.time()
                        _ = Statevector.from_instruction(qiskit_qft_qc)
                        end_time = time.time()
                        total_time_aer += (end_time - start_time)

                    avg_time_aer = total_time_aer / shots
                    results['qiskit_aer'].append((n, avg_time_aer))
                    print(f"Qiskit Statevector: Avg time over {shots} shots: {avg_time_aer:.6f} s")

                except Exception as e:
                    print(f"WARNING: Qiskit Statevector benchmark failed for N={n}. Error: {e}")

            # --- Benchmark 'Qiskit Opt'  ---
            elif method == 'qiskit_opt':
                total_time_opt = 0.0
                try:
                    qiskit_qft_qc = QiskitCircuit(n)
                    qiskit_qft_qc.append(qiskit_qft(n, do_swaps=True), range(n))
                    transpiled_qc = transpile(qiskit_qft_qc, optimization_level=3)

                    for i in range(shots):
                        start_time = time.time()
                        _ = Statevector.from_instruction(transpiled_qc)
                        end_time = time.time()
                        total_time_opt += (end_time - start_time)

                    avg_time_opt = total_time_opt / shots
                    results['qiskit_opt'].append((n, avg_time_opt))
                    print(f"Qiskit (L3 Opt): Avg time over {shots} shots: {avg_time_opt:.6f} s")

                except Exception as e:
                    print(f"WARNING: Qiskit Opt benchmark failed for N={n}. Error: {e}")

            # --- Benchmark 'Hybrid' (QuantumBackend) ---
            elif method == 'hybrid':
                total_time = 0.0

                # compile
                compiled_qc = hybrid_backend.optimise_circuit(base_qc)
                hybrid_backend.analyze_mode(base_qc)

                try:
                    for i in range(shots):
                        state_to_run = QuantumState(n)
                        state_to_run.basis_state(0)  # |0>

                        start_time = time.time()
                        hybrid_backend.execute(compiled_qc, initial_state=state_to_run, shots=1, inplace=True)
                        end_time = time.time()
                        total_time += (end_time - start_time)

                    avg_time = total_time / shots
                    results['hybrid'].append((n, avg_time))
                    print(f"Hybrid Backend: Avg time over {shots} shots: {avg_time:.6f} s")
                except Exception as e:
                    print(f"WARNING: Hybrid benchmark failed for N={n}. Error: {e}")

            # --- Benchmark 'dense', 'sparse', 'dense_opt', 'sparse_opt' ---
            else:
                use_opt = False
                actual_mode = method
                if method.endswith('_opt'):
                    use_opt = True
                    actual_mode = method.replace('_opt', '')

                qc_to_run = base_qc.copy()
                if use_opt:
                    qc_to_run.optimise()

                total_time = 0.0
                for i in range(shots):
                    state_to_run = QuantumState(n, mode=actual_mode)
                    state_to_run.basis_state(0)

                    if actual_mode == 'sparse':
                        state_to_run.to_sparse()

                    start_time = time.time()
                    qc_to_run.run(state_to_run, method=actual_mode)
                    end_time = time.time()
                    total_time += (end_time - start_time)

                avg_time = total_time / shots
                results[method].append((n, avg_time))
                print(f"{method.capitalize()}: Avg time over {shots} shots: {avg_time:.6f} s")

    return results


# ======================================================
#                          QPE
# ======================================================

def generate_qpe_circuit(n_estimation: int) -> QuantumCircuit:
    z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    target_qubit_index = n_estimation
    target_initial_state_gates = [QuantumGate.x(target_qubit_index)]

    return QuantumCircuit.qpe(
        t_qubits=n_estimation,
        unitary_matrix=z_matrix,
        m_qubits=1,
        target_initial_state_gates=target_initial_state_gates
    )


def benchmark_qpe(qubit_range: List[int], shots: int, methods_to_run: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Benchmarks the QPE circuit. Supports 'hybrid', '_opt', and 'qiskit_opt'.
    """
    results: Dict[str, List[Tuple[int, float]]] = {}
    for method in methods_to_run:
        results[method] = []

    hybrid_backend = QuantumBackend()

    for n in qubit_range:
        n_total = n + 1  # n estimation qubits + 1 target qubit
        print(f"\n--- Benchmarking QPE on {n} Estimation Qubits ({n_total} Total Qubits) ---")

        base_qc = generate_qpe_circuit(n)

        for method in methods_to_run:

            if method == 'qiskit_aer':
                total_time_aer = 0.0
                try:
                    qiskit_qpe_qc = QiskitCircuit(n_total)
                    qiskit_qpe_qc.append(PhaseEstimation(n, ZGate()), range(n_total))
                    qiskit_qpe_qc.x(n)

                    for i in range(shots):
                        start_time = time.time()
                        _ = Statevector.from_instruction(qiskit_qpe_qc)
                        end_time = time.time()
                        total_time_aer += (end_time - start_time)

                    avg_time_aer = total_time_aer / shots
                    results['qiskit_aer'].append((n, avg_time_aer))
                    print(f"Qiskit Statevector: Avg time over {shots} shots: {avg_time_aer:.6f} s")
                except Exception as e:
                    print(f"WARNING: Qiskit Statevector benchmark failed for N={n}. Error: {e}")

            elif method == 'qiskit_opt':
                total_time_opt = 0.0
                try:
                    qiskit_qpe_qc = QiskitCircuit(n_total)
                    qiskit_qpe_qc.append(PhaseEstimation(n, ZGate()), range(n_total))
                    qiskit_qpe_qc.x(n)
                    transpiled_qc = transpile(qiskit_qpe_qc, optimization_level=3)

                    for i in range(shots):
                        start_time = time.time()
                        _ = Statevector.from_instruction(transpiled_qc)
                        end_time = time.time()
                        total_time_opt += (end_time - start_time)

                    avg_time_opt = total_time_opt / shots
                    results['qiskit_opt'].append((n, avg_time_opt))
                    print(f"Qiskit (L3 Opt): Avg time over {shots} shots: {avg_time_opt:.6f} s")
                except Exception as e:
                    print(f"WARNING: Qiskit Opt benchmark failed for N={n}. Error: {e}")

            elif method == 'hybrid':
                total_time = 0.0

                # Pre-compile
                compiled_qc = hybrid_backend.optimise_circuit(base_qc)
                hybrid_backend.analyze_mode(base_qc)

                try:
                    for i in range(shots):
                        state_to_run = QuantumState(n_total)
                        state_to_run.basis_state(0)

                        start_time = time.time()
                        hybrid_backend.execute(compiled_qc, initial_state=state_to_run, shots=1, inplace=True)
                        end_time = time.time()
                        total_time += (end_time - start_time)
                    avg_time = total_time / shots
                    results['hybrid'].append((n, avg_time))
                    print(f"Hybrid Backend: Avg time over {shots} shots: {avg_time:.6f} s")
                except Exception as e:
                    print(f"WARNING: Hybrid benchmark failed for N={n}. Error: {e}")

            else:
                use_opt = False
                actual_mode = method
                if method.endswith('_opt'):
                    use_opt = True
                    actual_mode = method.replace('_opt', '')

                qc_to_run = base_qc.copy()
                if use_opt:
                    qc_to_run.optimise()

                total_time = 0.0
                for i in range(shots):
                    state_to_run = QuantumState(n_total, mode=actual_mode)
                    state_to_run.basis_state(0)
                    if actual_mode == 'sparse':
                        state_to_run.to_sparse()

                    start_time = time.time()
                    qc_to_run.run(state_to_run, method=actual_mode)
                    end_time = time.time()
                    total_time += (end_time - start_time)

                avg_time = total_time / shots
                results[method].append((n, avg_time))
                print(f"{method.capitalize()}: Avg time over {shots} shots: {avg_time:.6f} s")

    return results


# ======================================================
#                         GROVERS
# ======================================================

def _build_qiskit_grover(n_qubits: int, marked_index: int, iterations: int) -> QiskitCircuit:
    """ Helper to build a Qiskit circuit matching custom Grover implementation. """
    qc = QiskitCircuit(n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    bin_str = format(marked_index, f'0{n_qubits}b')

    for _ in range(iterations):
        for q in qubits:
            if bin_str[n_qubits - 1 - q] == '0':
                qc.x(q)
        if n_qubits > 1:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        else:
            qc.z(0)
        for q in qubits:
            if bin_str[n_qubits - 1 - q] == '0':
                qc.x(q)
        qc.h(qubits)
        qc.x(qubits)
        if n_qubits > 1:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        else:
            qc.z(0)
        qc.x(qubits)
        qc.h(qubits)
    return qc


def benchmark_grover(qubit_range: List[int], shots: int, methods_to_run: List[str]) -> Dict[
    str, List[Tuple[int, float]]]:
    """
    Benchmarks Grover's Search Algorithm. Supports 'hybrid', '_opt', and 'qiskit_opt'.
    """
    results: Dict[str, List[Tuple[int, float]]] = {}
    for method in methods_to_run:
        results[method] = []

    hybrid_backend = QuantumBackend()

    for n in qubit_range:
        print(f"\n--- Benchmarking Grover's Search on {n} Qubits ---")

        marked_index = int('10' * (n // 2) + '1' * (n % 2), 2)
        N = 2 ** n
        R = round(math.pi / 4 * math.sqrt(N))
        print(f"Target Index: {marked_index}, Iterations: {R}")

        base_qc = QuantumCircuit.grover_search(n, marked_index)

        for method in methods_to_run:

            if method == 'qiskit_aer':
                total_time_aer = 0.0
                try:
                    qiskit_qc = _build_qiskit_grover(n, marked_index, R)
                    for _ in range(shots):
                        start_time = time.time()
                        _ = Statevector.from_instruction(qiskit_qc)
                        total_time_aer += (time.time() - start_time)
                    avg_time_aer = total_time_aer / shots
                    results['qiskit_aer'].append((n, avg_time_aer))
                    print(f"Qiskit Statevector: Avg time over {shots} shots: {avg_time_aer:.6f} s")
                except Exception as e:
                    print(f"WARNING: Qiskit benchmark failed for N={n}. Error: {e}")

            elif method == 'qiskit_opt':
                total_time_opt = 0.0
                try:
                    qiskit_qc = _build_qiskit_grover(n, marked_index, R)
                    transpiled_qc = transpile(qiskit_qc, optimization_level=3)

                    for _ in range(shots):
                        start_time = time.time()
                        _ = Statevector.from_instruction(transpiled_qc)
                        total_time_opt += (time.time() - start_time)
                    avg_time_opt = total_time_opt / shots
                    results['qiskit_opt'].append((n, avg_time_opt))
                    print(f"Qiskit (L3 Opt): Avg time over {shots} shots: {avg_time_opt:.6f} s")
                except Exception as e:
                    print(f"WARNING: Qiskit Opt benchmark failed for N={n}. Error: {e}")

            elif method == 'hybrid':
                total_time = 0.0

                compiled_qc = hybrid_backend.optimise_circuit(base_qc)
                hybrid_backend.analyze_mode(base_qc)

                try:
                    for _ in range(shots):
                        state_to_run = QuantumState(n)
                        state_to_run.basis_state(0)

                        start_time = time.time()
                        hybrid_backend.execute(compiled_qc, initial_state=state_to_run, shots=1, inplace=True)
                        end_time = time.time()
                        total_time += (end_time - start_time)
                    avg_time = total_time / shots
                    results['hybrid'].append((n, avg_time))
                    print(f"Hybrid Backend: Avg time over {shots} shots: {avg_time:.6f} s")
                except Exception as e:
                    print(f"WARNING: Hybrid benchmark failed for N={n}. Error: {e}")

            else:
                use_opt = False
                actual_mode = method
                if method.endswith('_opt'):
                    use_opt = True
                    actual_mode = method.replace('_opt', '')

                qc_to_run = base_qc.copy()
                if use_opt:
                    qc_to_run.optimise()

                total_time = 0.0
                for _ in range(shots):
                    state_to_run = QuantumState(n, mode=actual_mode)
                    if actual_mode == 'sparse':
                        state_to_run.to_sparse()

                    start_time = time.time()
                    qc_to_run.run(state_to_run, method=actual_mode)
                    end_time = time.time()
                    total_time += (end_time - start_time)

                avg_time = total_time / shots
                results[method].append((n, avg_time))
                print(f"{method.capitalize()}: Avg time over {shots} shots: {avg_time:.6f} s")

    return results


# ======================================================
#                          GHZ
# ======================================================

def generate_ghz_circuit(n_qubits: int) -> QuantumCircuit:
    """ Generates a GHZ state circuit. """
    qc = QuantumCircuit(n_qubits)
    qc.add_gate(QuantumGate.h(0))
    for i in range(n_qubits - 1):
        qc.add_gate(QuantumGate.cx(i, i + 1))
    return qc


def benchmark_ghz_circuit(qubit_range: List[int], shots: int, methods_to_run: List[str]) -> Dict[
    str, List[Tuple[int, float]]]:
    """
    Benchmarks the GHZ circuit. Supports 'hybrid', '_opt', and 'qiskit_opt'.
    """
    results: Dict[str, List[Tuple[int, float]]] = {}
    for method in methods_to_run:
        results[method] = []

    hybrid_backend = QuantumBackend()

    for n in qubit_range:
        print(f"\n--- Benchmarking GHZ (Sparse Advantage) on {n} Qubits ---")

        base_qc = generate_ghz_circuit(n)

        for method in methods_to_run:
            if method == 'qiskit_aer':
                try:
                    qiskit_qc = QiskitCircuit(n)
                    qiskit_qc.h(0)
                    for i in range(n - 1):
                        qiskit_qc.cx(i, i + 1)

                    total_time = 0.0
                    for _ in range(shots):
                        start = time.time()
                        _ = Statevector.from_instruction(qiskit_qc)
                        total_time += (time.time() - start)

                    avg_time = total_time / shots
                    results['qiskit_aer'].append((n, avg_time))
                    print(f"Qiskit: Avg time: {avg_time:.6f} s")
                except Exception as e:
                    print(f"Qiskit failed for N={n}: {e}")

            elif method == 'qiskit_opt':
                try:
                    qiskit_qc = QiskitCircuit(n)
                    qiskit_qc.h(0)
                    for i in range(n - 1):
                        qiskit_qc.cx(i, i + 1)
                    transpiled_qc = transpile(qiskit_qc, optimization_level=3)
                    total_time = 0.0
                    for _ in range(shots):
                        start = time.time()

                        _ = Statevector.from_instruction(transpiled_qc)
                        total_time += (time.time() - start)

                    avg_time = total_time / shots
                    results['qiskit_opt'].append((n, avg_time))
                    print(f"Qiskit (L3 Opt): Avg time: {avg_time:.6f} s")
                except Exception as e:
                    print(f"Qiskit Opt failed for N={n}: {e}")

            elif method == 'hybrid':
                total_time = 0.0

                compiled_qc = hybrid_backend.optimise_circuit(base_qc)
                hybrid_backend.analyze_mode(base_qc)

                try:
                    for _ in range(shots):
                        state_to_run = QuantumState(n)
                        state_to_run.basis_state(0)

                        start_time = time.time()
                        hybrid_backend.execute(compiled_qc, initial_state=state_to_run, shots=1, inplace=True)
                        total_time += (time.time() - start_time)

                    avg_time = total_time / shots
                    results['hybrid'].append((n, avg_time))
                    print(f"Hybrid Backend: Avg time over {shots} shots: {avg_time:.6f} s")
                except Exception as e:
                    print(f"WARNING: Hybrid benchmark failed for N={n}. Error: {e}")

            else:
                use_opt = False
                actual_mode = method
                if method.endswith('_opt'):
                    use_opt = True
                    actual_mode = method.replace('_opt', '')

                qc_to_run = base_qc.copy()
                if use_opt:
                    qc_to_run.optimise()

                total_time = 0.0
                for _ in range(shots):
                    state_to_run = QuantumState(n, mode=actual_mode)
                    if actual_mode == 'sparse':
                        state_to_run.to_sparse()

                    start_time = time.time()
                    qc_to_run.run(state_to_run, method=actual_mode)
                    total_time += (time.time() - start_time)

                avg_time = total_time / shots
                results[method].append((n, avg_time))
                print(f"{method.capitalize()}: Avg time: {avg_time:.6f} s")

    return results