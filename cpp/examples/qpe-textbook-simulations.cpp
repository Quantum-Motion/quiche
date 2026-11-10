/*
 * Copyright 2026 Quantum Motion Technologies Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file
 * Demo of textbook (multi-ancilla) quantum phase estimation using each Hamiltonian simulation
 * method: Trotterisation, QDRIFT, qubitisation and optimised qubitisation, for a minimal basis
 * set H2 molecule.
 *
 * For each number of QPE ancillas the demo prints the register size, the estimated energy,
 * its error and the simulation run time.
 *
 * Trade-offs:
 * - With enough ancillas the error is set by the phase grid 1 / 2^n rather than by the simulation
 *   method. The Trotter energy is the same for 6 and 8 ancillas because the true phase, about
 *   -0.2203, rounds to the same grid point -14/64 = -56/256 (error 0.0075). Textbook QPE with n
 *   ancillas extracts the same n bits as iterative QPE with n bits, only in parallel on n ancillas
 *   rather than sequentially on one, so both give the same answer.
 * - QDRIFT matches Trotterisation when its random compilation lands on the same grid point (6
 *   ancillas), but a single random circuit can shift the most likely outcome to a neighbouring
 *   point (errors 0.155 and 0.011 for 4 and 8 ancillas). On hardware this is averaged out by
 *   drawing a new circuit for every shot.
 * - Qubitisation needs ceil(log2(numTerms)) = 4 extra ancillas but no evolution time. Its walk
 *   operator has eigenphases +-arccos(E / lambda), so a phase error is amplified by
 *   2 pi lambda sin(2 pi phase), about 9.9 here. Its true phase, about 0.3428, also rounds to the
 *   same grid point 22/64 = 88/256 with 6 and 8 ancillas, so the energy (error 0.0089) is unchanged.
 * - Optimised qubitisation gives the same energy in about half the simulation time, as it only
 *   controls the reflections and applies 2^(k-1) uncontrolled walk steps instead of 2^k controlled
 *   ones (Babbush et al., arXiv:1805.03662).
 * - Simulation time is not hardware cost: Trotterisation is the slowest to simulate because each
 *   of its rotations is applied in turn, whereas the cost on a quantum computer is measured in
 *   T gates and circuit depth, which the Python resource estimation backend computes.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <quest.h>
#include <quest/src/core/constants.hpp>

#include "quiche/mappings.hpp"
#include "quiche/qdrift.hpp"
#include "quiche/qpe.hpp"
#include "quiche/quest-patches.hpp"
#include "quiche/utils.hpp"

namespace {

// Contiguous register of `size` qubits starting at qubit `start`
std::vector<int> getRegister(int start, int size) {
    std::vector<int> reg(size);
    std::iota(reg.begin(), reg.end(), start);
    return reg;
}

void printRow(const std::string &method, int numQubits, double energy, double exactEnergy,
              std::chrono::steady_clock::time_point start) {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::left << std::setw(20) << method << std::setw(8) << numQubits << std::setw(12) << energy
              << std::setw(12) << std::abs(energy - exactEnergy) << ms << '\n';
}

} // namespace

int main() {
    initQuESTEnv();

    // Hamiltonian
    const char *hydrogenMinimalJWStr = R"(
    -0.098864 IIII
    +0.171198 IIIZ
    +0.171198 IIZI
    -0.222786 IZII
    -0.222786 ZIII
    +0.168622 IIZZ
    +0.045322 YXXY
    -0.045322 XXYY
    -0.045322 YYXX
    +0.045322 XYYX
    +0.120545 IZIZ
    +0.165867 ZIIZ
    +0.165867 IZZI
    +0.120545 ZIZI
    +0.174348 ZZII
    )";
    int numElectrons = 2;
    int numDataQubits = 4;
    double exactEnergy = -1.137272; // exact diagonalisation

    PauliStrSum rawHamiltonian = createInlinePauliStrSum(hydrogenMinimalJWStr);
    qcomp idCoeff = getIdentityCoeff(rawHamiltonian);
    PauliStrSum hamiltonian = cloneWithoutIdentity(rawHamiltonian);

    qindex hf = mappings::getHartreeFockStateJW(numElectrons);

    // lambda bounds every eigenvalue, |E| <= lambda, and scales the qubitisation block encoding.
    // The qubitisation register indexes the terms of the Hamiltonian.
    qreal lambda = qdrift::getPauliStrSumNorm(hamiltonian);
    int numQubitisationAncillas = std::ceil(std::log2(hamiltonian.numTerms));

    // Eigenphases E t / 2 pi span a width of 2 lambda t / 2 pi, which must fit in one period to
    // avoid aliasing. The factor 1.25 leaves headroom, as in cpp/tests/qpe.cpp.
    double t = 2 * const_PI / (1.25 * 2 * lambda);

    // Trotter and QDRIFT parameters
    int order = 2;
    int trotterReps = 10;
    int qdriftReps = 100;
    std::mt19937_64 rng(965456);

    std::cout << std::fixed << std::setprecision(6);

    for (int numQPEAncillas = 4; numQPEAncillas <= 8; numQPEAncillas += 2) {

        // Data register: [0, numDataQubits)
        // QPE register: [numDataQubits, numDataQubits + numQPEAncillas)
        // Qubitisation register: after the QPE register (qubitised methods only)
        std::vector<int> qpeAncillas = getRegister(numDataQubits, numQPEAncillas);
        std::vector<int> qubitisationAncillas = getRegister(numDataQubits + numQPEAncillas, numQubitisationAncillas);

        std::cout << "\n---------- " << numQPEAncillas << " QPE ancillas ----------\n"
                  << "QPE: " << formatVector(qpeAncillas) << "\n"
                  << "Qubitisation: " << formatVector(qubitisationAncillas) << "\n"
                  << "method              qubits  energy      |err|       time (ms)\n";

        // Each method gets a fresh Qureg sized for its own registers, destroyed once its row is printed.
        // Trotterisation approximates U = exp(iHt), so the phase is E t / 2 pi.
        {
            auto start = std::chrono::steady_clock::now();
            Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
            initClassicalState(qureg, hf);
            qreal phase = qpe::getPhaseTextbookTrotter(qureg, hamiltonian, qpeAncillas, order, trotterReps, t);
            qreal energy = qpe::getEnergyFromTrotterPhase(phase, t, idCoeff.real());
            printRow("Trotter", qureg.numQubits, energy, exactEnergy, start);
            destroyQureg(qureg);
        }

        // QDRIFT approximates the same U by random sampling, so it shares the phase-to-energy map.
        // The generator is shared across rows, so each row uses a different random compilation.
        {
            auto start = std::chrono::steady_clock::now();
            Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
            initClassicalState(qureg, hf);
            qreal phase = qpe::getPhaseTextbookQDRIFT(qureg, hamiltonian, qpeAncillas, qdriftReps, t, rng);
            qreal energy = qpe::getEnergyFromTrotterPhase(phase, t, idCoeff.real());
            printRow("QDRIFT", qureg.numQubits, energy, exactEnergy, start);
            destroyQureg(qureg);
        }

        // The qubitisation walk operator has eigenphases +-arccos(E / lambda), so there is no
        // evolution time and the energy is recovered as lambda cos(2 pi phase)
        {
            auto start = std::chrono::steady_clock::now();
            Qureg qureg = createQureg(numDataQubits + numQPEAncillas + numQubitisationAncillas);
            initClassicalState(qureg, hf);
            qreal phase = qpe::getPhaseTextbookQubitised(qureg, hamiltonian, qpeAncillas, qubitisationAncillas);
            qreal energy = qpe::getEnergyFromQubitisationPhase(phase, lambda, idCoeff.real());
            printRow("Qubitisation", qureg.numQubits, energy, exactEnergy, start);
            destroyQureg(qureg);
        }

        // The optimised walk only controls the reflections and halves the number of walk steps,
        // giving the same phase with fewer controlled operations
        {
            auto start = std::chrono::steady_clock::now();
            Qureg qureg = createQureg(numDataQubits + numQPEAncillas + numQubitisationAncillas);
            initClassicalState(qureg, hf);
            qreal phase =
                qpe::getPhaseTextbookQubitisedOptimised(qureg, hamiltonian, qpeAncillas, qubitisationAncillas);
            qreal energy = qpe::getEnergyFromQubitisationPhase(phase, lambda, idCoeff.real());
            printRow("Qubitisation-Opt", qureg.numQubits, energy, exactEnergy, start);
            destroyQureg(qureg);
        }
    }

    destroyPauliStrSum(rawHamiltonian);
    destroyPauliStrSum(hamiltonian);
    finalizeQuESTEnv();
    return 0;
}
