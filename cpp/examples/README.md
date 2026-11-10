# C++ examples

Standalone programs that use the QUICHE C++ library and [QuEST](https://github.com/QuEST-Kit/QuEST) to simulate quantum phase estimation (QPE) for ground-state energies. Each example is a single `.cpp` file. All of them use the same minimal-basis H2 Hamiltonian under the Jordan–Wigner mapping, which has an exact ground-state energy of −1.137272 Ha.

## Building and running

The examples are built only when `QUICHE_BUILD_EXAMPLES` is on:

```bash
cmake -B build-cpp -D QUICHE_BUILD_EXAMPLES=ON
cmake --build build-cpp -j 8
./build-cpp/cpp/examples/qpe-textbook
```

To build a single example, pass `--target <name>` to `cmake --build`.

## Overview

### Single-method demos

Each of these runs one QPE algorithm with one Hamiltonian simulation method and prints the estimated phase and energy.

| Example | QPE algorithm | Hamiltonian simulation |
|---|---|---|
| [`qpe-textbook`](qpe-textbook.cpp) | Textbook (multi-ancilla, inverse QFT) | Trotterisation |
| [`qpe-naive`](qpe-naive.cpp) | Naive (single ancilla) | QDRIFT |
| [`qpe-iterative`](qpe-iterative.cpp) | Iterative (single ancilla) | Trotterisation |
| [`qpe-kitaev`](qpe-kitaev.cpp) | Kitaev (single ancilla) | QDRIFT |
| [`qpe-qubitised`](qpe-qubitised.cpp) | Textbook | Qubitisation |
| [`qpe-qubitised-optimised`](qpe-qubitised-optimised.cpp) | Textbook | Qubitisation with the optimised walk of [Babbush et al.](https://arxiv.org/abs/1805.03662) |
| [`qdrift`](qdrift.cpp) | — | QDRIFT vs Trotterisation time evolution for a random Hamiltonian |

### Comparisons

These run several methods side by side and print tables. The header comment of each file summarises the results and explains where each method performs best.

| Example | Compares | Run time |
|---|---|---|
| [`hamiltonian-simulation`](hamiltonian-simulation.cpp) | Trotterisation, QDRIFT and qubitisation: time-evolution fidelity at equal cost (Pauli rotations), for H2 and for a 200-term Hamiltonian, then inside textbook QPE | ~4 s |
| [`qpe-single-ancilla`](qpe-single-ancilla.cpp) | Naive, iterative and Kitaev QPE: precision vs number of bits, the evolution-time limit of naive QPE, and spread over QDRIFT seeds | ~3 s |
| [`qpe-textbook-simulations`](qpe-textbook-simulations.cpp) | Textbook QPE with Trotterisation, QDRIFT, qubitisation and optimised qubitisation for 4, 6 and 8 ancillas: qubit count, energy error and simulation time | ~10 s |

Run times were measured on an Apple Silicon laptop.

The main findings:

- **Trotterisation** wins for Hamiltonians with few terms of similar size, such as H2.
- **QDRIFT** wins for Hamiltonians with many terms dominated by a few large ones, at low precision, because its cost does not depend on the number of terms.
- **Qubitisation** has no time-step error and needs no evolution time. It uses ⌈log₂(numTerms)⌉ extra ancillas, and its walk steps are not directly comparable to Pauli rotations. The T-gate counts from the Python resource estimation backend give a fair comparison.
- **Single-ancilla methods** trade circuit width for depth. Iterative QPE gains about one bit of precision per round. Kitaev and naive QPE look very accurate here because the simulator reads ancilla expectation values exactly, which on hardware would need many shots.
