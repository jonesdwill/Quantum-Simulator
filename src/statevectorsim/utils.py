import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from typing import List, Dict, Union

def single_qubit_bloch_vector(state, qubit_index):
    """
    Compute the Bloch vector (x, y, z) for a specific qubit in a multi-qubit state.
    """
    n_qubits = int(np.log2(len(state)))
    state_tensor = state.reshape([2] * n_qubits)

    axes_to_trace = tuple(i for i in range(n_qubits) if i != qubit_index)

    rho = np.tensordot(state_tensor, np.conj(state_tensor),
                       axes=(axes_to_trace, axes_to_trace))

    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])

    vec = np.array([x, y, z])

    return vec


def plot_bloch_spheres(state, fig_size=(4, 4), max_cols=4):
    """
    Bloch sphere plot with:
    - X, Y, Z equators
    - |0⟩ and |1⟩ labels
    - No axis arrows
    - No grid
    - Minimal clean aesthetic
    """

    n_qubits = int(np.log2(len(state)))

    n_cols = min(n_qubits, max_cols)
    n_rows = math.ceil(n_qubits / max_cols)

    fig = plt.figure(figsize=(fig_size[0] * n_cols, fig_size[1] * n_rows))

    # Sphere surface mesh
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    # Equators
    eq_u = np.linspace(0, 2*np.pi, 200)
    eq_zero = np.zeros_like(eq_u)

    # XY equator (Z=0)
    eq_x = np.cos(eq_u)
    eq_y = np.sin(eq_u)
    eq_z = eq_zero

    # XZ equator (Y=0)
    eq2_x = np.cos(eq_u)
    eq2_y = eq_zero
    eq2_z = np.sin(eq_u)

    # YZ equator (X=0)
    eq3_x = eq_zero
    eq3_y = np.cos(eq_u)
    eq3_z = np.sin(eq_u)

    for q in range(n_qubits):

        vec = single_qubit_bloch_vector(state, q)

        ax = fig.add_subplot(n_rows, n_cols, q+1, projection='3d')

        # Sphere surface
        ax.plot_surface(
            xs, ys, zs,
            rstride=1, cstride=1,
            color="white", alpha=0.05,
            edgecolor="gray", linewidth=0.3
        )

        ax.plot(eq_x, eq_y, eq_z, color='black', linewidth=0.6)   # XY equator
        ax.plot(eq2_x, eq2_y, eq2_z, color='black', linewidth=0.6) # XZ equator
        ax.plot(eq3_x, eq3_y, eq3_z, color='black', linewidth=0.6) # YZ equator

        ax.plot([-1, 1], [0, 0], [0, 0], color="black", linewidth=0.8)  # X-axis
        ax.plot([0, 0], [-1, 1], [0, 0], color="black", linewidth=0.8)  # Y-axis
        ax.plot([0, 0], [0, 0], [-1, 1], color="black", linewidth=0.8)  # Z-axis

        # plot vector
        ax.quiver(
            0, 0, 0,  # start at origin
            vec[0], vec[1], vec[2],  # vector components
            color='blue',
            linewidth=2,
            arrow_length_ratio=0.2,
            linestyle='-',  # solid line
            alpha=0.9
        )

        ## --- LABEL & FORMAT ---
        # Inside your loop over qubits:
        n_qubits = int(np.log2(len(state)))
        state_tensor = state.reshape([2] * n_qubits)

        # Reduced density matrix for qubit q
        axes_to_trace = tuple(i for i in range(n_qubits) if i != q)
        rho = np.tensordot(state_tensor, np.conj(state_tensor),
                           axes=(axes_to_trace, axes_to_trace))

        # Compute amplitudes for labeling
        alpha = np.sqrt(np.real(rho[0, 0]))
        beta = np.sqrt(np.real(rho[1, 1]))
        phase = np.angle(rho[0, 1])
        beta = beta * np.exp(1j * phase)

        title_str = f"Qubit {q}\n$|\\psi\\rangle = {alpha:.2f}|0\\rangle + {abs(beta):.2f}|1\\rangle$"
        ax.set_title(title_str, fontsize=10)

        ax.text(0, 0, 1.5, r"$|0\rangle$", ha='center', va='center', fontsize=10)
        ax.text(0, 0, -1.5, r"$|1\rangle$", ha='center', va='center', fontsize=10)

        label_offset = 1.15
        ax.text(label_offset, 0, 0, 'X', ha='center', va='center', fontsize=10)
        ax.text(0, label_offset, 0, 'Y', ha='center', va='center', fontsize=10)
        ax.text(0, 0, label_offset, 'Z', ha='center', va='center', fontsize=10)

        ax.set_box_aspect([1, 1, 1])

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.set_axis_off()
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.view_init(elev=25, azim=40)

    plt.tight_layout()
    plt.show()


def statevector_to_dataframe(state: np.ndarray, little_endian=True):
    """
    Convert a statevector to a pandas DataFrame.
    Little-endian (default) means qubit 0 = LSB (rightmost bit).

    Returns a DataFrame with columns: Index, State, Amplitude
    """
    n = int(np.log2(len(state)))

    indices = np.arange(len(state))

    # Binary strings for basis states
    if little_endian:
        states = [f"|{i:0{n}b}>" for i in indices]
    else:
        # Big-endian: reverse bits
        states = [f"|{format(i, f'0{n}b')[::-1]}>" for i in indices]

    # Format amplitudes
    amplitudes = [f"{amp.real:.4f}{'+' if amp.imag >= 0 else '-'}{abs(amp.imag):.4f}j" for amp in state]

    df = pd.DataFrame({
        "Index": indices,
        "State": states,
        "Amplitude": amplitudes
    })

    return df


def plot_histogram(results: Dict[str, int], shots: int):
    """
    Generates and displays a probability histogram from measurement results.
    """
    if not results:
        print("No results to plot.")
        return

    # Sort keys for consistent plotting order (e.g., '00', '01', '10', '11')
    outcomes = sorted(results.keys())
    counts = [results.get(o, 0) for o in outcomes]
    probabilities = [c / shots for c in counts]

    # Create plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(outcomes, probabilities, alpha=0.7)

    # Add title and labels
    plt.title(f'Histogram of Monte-Carlo Simulation ({shots} Shots)', fontsize=14)
    plt.xlabel('Measurement Outcome', fontsize=12)
    plt.ylabel('Probability', fontsize=12)

    # Set y-axis limits and show a grid
    plt.ylim(0, max(probabilities) * 1.1 or 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add probability text labels on top of the bars
    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_val + 0.01,
                 f'{y_val:.3f}', ha='center', va='bottom')

    plt.show()
