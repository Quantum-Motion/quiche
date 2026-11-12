# C++ examples

Standalone programs that use the QUICHE C++ library and [QuEST](https://github.com/QuEST-Kit/QuEST) to demonstrate Hamiltonian simulation methods and quantum phase estimation (QPE) algorithms.

## Building and running

The examples are built only when `QUICHE_BUILD_EXAMPLES` is on. From the repository root:

```bash
cmake -B build -D QUICHE_BUILD_EXAMPLES=ON
cmake --build build
./build/cpp/examples/qpe-textbook
```

To build a single example, pass `--target <name>` to `cmake --build`.

## Overview

### Single-method demos

Each QPE demo runs one QPE algorithm with one Hamiltonian simulation method on a minimal basis set H2 molecule and prints the estimated phase and energy.

| Example | Algorithm (QPE + Hamiltonian simulation) |
|---|---|
| [`qpe-textbook`](qpe-textbook.cpp) | Textbook QPE (multi-ancilla, inverse QFT) + Trotterisation |
| [`qpe-naive`](qpe-naive.cpp) | Naive QPE (single ancilla) + QDRIFT |
| [`qpe-iterative`](qpe-iterative.cpp) | Iterative QPE (single ancilla) + Trotterisation |
| [`qpe-kitaev`](qpe-kitaev.cpp) | Kitaev QPE (single ancilla) + QDRIFT |
| [`qpe-qubitised`](qpe-qubitised.cpp) | Textbook QPE + qubitisation |
| [`qpe-qubitised-optimised`](qpe-qubitised-optimised.cpp) | Textbook QPE + qubitisation with the optimised walk of [Babbush et al.](https://arxiv.org/abs/1805.03662) |
| [`qdrift`](qdrift.cpp) | No QPE: QDRIFT vs Trotterisation time evolution for a random Hamiltonian |

### Comparisons

These run different methods side by side and compare their performance.

| Example | Compares |
|---|---|
| [`hamiltonian-simulation`](hamiltonian-simulation.cpp) | Trotterisation, QDRIFT and qubitisation: time-evolution fidelity against cost (Pauli rotations), for H2 and for a random 200-term Hamiltonian, then inside textbook QPE |
| [`qpe-single-ancilla`](qpe-single-ancilla.cpp) | Naive, iterative and Kitaev QPE: precision vs number of bits, the evolution-time limit of naive QPE, and spread over QDRIFT seeds |
| [`qpe-textbook-simulations`](qpe-textbook-simulations.cpp) | Textbook QPE with Trotterisation, QDRIFT, qubitisation and optimised qubitisation for 4, 6 and 8 ancillas: qubit count, energy error and simulation time |

The main findings:

- **Trotterisation** wins for Hamiltonians with few terms of similar size, such as H2.
- **QDRIFT** wins for Hamiltonians with many terms dominated by a few large ones, at low precision, because its cost does not depend on the number of terms.
- **Qubitisation** has no time-step error and needs no evolution time. It uses ⌈log₂(numTerms)⌉ extra ancillas, and its walk steps are not directly comparable to Pauli rotations. The T-gate counts from the Python resource estimation backend give a fair comparison.
- **Single-ancilla methods** trade circuit width for the number of circuit runs. Each run uses one ancilla and a shorter circuit than textbook QPE (no inverse QFT, and at most 2^(n−1) applications of U rather than 2^n − 1), but the phase is built up over many runs. Iterative QPE gains about one bit of precision per round. Kitaev QPE looks accurate from only 2 bits because the simulator reads ancilla expectation values exactly. On hardware, naive and Kitaev QPE would need many shots to estimate them.
