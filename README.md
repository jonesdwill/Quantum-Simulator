# Quantum Simulator

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![PyPI version](https://img.shields.io/badge/version-0.1.0-blue)

A lightweight, quantum state-vector circuit simulator, written in Python.

---

## Features

* **`QuantumState`**: Manages the state vector, handles single-qubit initialization, state display, and state-collapse measurement.
* **`QuantumGate`**: Comprehensive collection of single-qubit, rotation, controlled, and multi-controlled gates applied to QuantumState. Functionality to implement any unitary gate.
* **`QuantumCircuit`**: Build and run series of quantum gates. Run Monte Carlo simulations. In-built standard quantum circuits. 
* **`utils`**: Helper functions for visualization, including Bloch sphere plots and probability histograms.
* **`test_suite`**: A dedicated file for functional testing of all implemented gates.

---

## Installation

To set-up, follow the steps:

1. Clone the repository:

```
git clone https://github.com/jonesdwill/Quantum-Simulator.git
cd Quantum-Simulator
```

2. Set up a virtual environment (reccomended): If you don't have it already, install **`venv`**:
```
py -m pip install venv
```
Navigate to project directory. Create and activate virtual environment.
```
py -m venv venv
venv\Scripts\activate
```
3. Install Required Packages
```
pip install -r requirements.txt
```
---
## Gate Look-up (`QuantumGate`)

The `QuantumGate` class includes an array of static methods for constructing common single-qubit and multi-qubit gates.

| Category | Gate                             | Method Signature | Description |
| :--- |:---------------------------------| :--- | :--- |
| **Standard** | Pauli-X (NOT)                    | `x(targets)` | Bit flip. |
| | Pauli-Y                          | `y(targets)` | Phase and bit flip. |
| | Pauli-Z                          | `z(targets)` | Phase flip on $|1\rangle$. |
| | Hadamard                         | `h(targets)` | Creates superposition. |
| | Identity                         | `i(targets)` | No change. |
| **Phase** | Phase (S)                        | `s(targets)` | $Z$ rotation by $\pi/2$. |
| | Inverse Phase ($S^\dagger$)      | `sdg(targets)` | $Z$ rotation by $-\pi/2$. |
| | $\pi/8$ (T)                      | `t(targets)` | $Z$ rotation by $\pi/4$. |
| | Inverse $\pi/8$ ($T^\dagger$)    | `tdg(targets)` | $Z$ rotation by $-\pi/4$. |
| **Rotation** | $R_x(\theta)$                    | `rx(targets, theta)` | Rotation about X-axis. |
| | $R_y(\theta)$                    | `ry(targets, theta)` | Rotation about Y-axis. |
| | $R_z(\theta)$                    | `rz(targets, theta)` | Rotation about Z-axis. |
| **Controlled** | CNOT (CX)                        | `cx(control, target)` | Controlled-X. |
| | Controlled-Y (CY)                | `cy(control, target)` | Controlled-Y. |
| | Controlled-Z (CZ)                | `cz(control, target)` | Controlled-Z. |
| | SWAP                             | `swap(q1, q2)` | Swaps two qubit states. |
| | Controlled- $R_x(\theta)$  (CRX) | `crx(c, t, theta)` | Controlled-Rotation-X. |
| | Controlled- $R_y(\theta)$  (CRY) | `cry(c, t, theta)` | Controlled-Rotation-Y. |
| | Controlled- $R_z(\theta)$  (CRZ) | `crz(c, t, theta)` | Controlled-Rotation-Z. |
| **Multi-Controlled**| Toffoli (CCX/MCX)                | `mcx(controls, target)`| Multi-Controlled X. |
| | MCY                              | `mcy(controls, target)`| Multi-Controlled Y. |
| | MCZ                              | `mcz(controls, target)`| Multi-Controlled Z. |

---

## Circuit Look-up (`QuantumCircuit`)

The `QuantumCircuit` class includes static methods to quickly generate common n-qubit circuits.

| Circuit Name |                                 Method Signature                                 | Qubit Count |                                                    Description                                                     |
| :---: |:--------------------------------------------------------------------------------:| :---: |:------------------------------------------------------------------------------------------------------------------:|
| **Bell State** |                        `QuantumCircuit.bell(n_qubits: 2)`                        | 2 |                     Creates the Bell state $(\vert 00 \rangle + \vert 11 \rangle)/ \sqrt{2}$.                      |
| **GHZ State** |                       `QuantumCircuit.ghz(n_qubits: int)`                        | $\geq 2$ | Creates the $n$-qubit Greenberger–Horne–Zeilinger state $(\vert 00..00 \rangle + \vert 11..11 \rangle)/ \sqrt{2}$. |
| **Grover Search** |               `QuantumCircuit.grover_search(n_qubits, marked_idx)`               | $\geq 2$ |                                   Grover's algorithm to find marked state index.                                   |
| **QFT** |            `QuantumCircuit.qft(qubits: List[int], swap_endian: bool)`            | $\geq 1$ |                             Performs Quantum Fourier Transform (QFT) on QuantumState.                              |
| **IQFT** | `QuantumCircuit.qft(qubits: List[int], swap_endian: bool, inverse: bool = True)` | $\geq 1$ |                         Performs Inverse Quantum Fourier Transform (IQFT) on QuantumState.                         |
| **QFT Adder** |                    `QuantumCircuit.qft_adder(n_qubits: int)`                     | $2n$ |       Performs addition: $\vert A \rangle \vert B \rangle \to \vert A \rangle \vert B+A \pmod{2^n} \rangle$.       |
## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

---

## Version

Current version: 0.1.0
