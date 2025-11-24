# Quantum Simulator

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![PyPI version](https://img.shields.io/badge/version-0.1.0-blue)

A lightweight, quantum state-vector circuit simulator, written in Python.

---

## Core Components

### QuantumState
State vector of an \(n\)-qubit system. Includes:
- Initialising basis states  
- Measurement collapse of individual or all qubits.
- Efficient application of gates using bitmask indexing.  

### QuantumGate
Defines quantum gates and how they act on specific qubits. Includes:
- Standard single-qubit gates: X, Y, Z, H.
- Controlled gates: CX.
- Multi-controlled gates: MCX.
- Gate construction from arbitrary unitary matrices.

### QuantumCircuit
Ordered list of quantum gates that can be executed on a QuantumState. Supports:
- Adding gates.
- Execution. 
- Decoupled circuit construction from state evolution  

### Utils
A collection of supporting functions used internally. 

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jonesdwill/Quantum-Simulator.git
cd Quantum-Simulator
```

I recommend setting up a virtual environment to use as your interpreter. If you don't have it already, install venv:
```
py -m pip install venv
```
Navigate to project directory, or wherever you want to store your virtual environment:
```
cd path\to\project
```
Create virtual environment and install packages:
```
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

---

## Version

Current version: 0.1.0
