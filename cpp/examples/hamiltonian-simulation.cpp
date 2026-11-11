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
 * Demo comparing Hamiltonian simulation methods: Trotterisation, QDRIFT and qubitisation.
 *
 * Part 1 compares the accuracy of time evolution for Trotterisation and QDRIFT at equal cost,
 * counted as the number of Pauli rotations, for a minimal basis set H2 molecule and for a
 * random Hamiltonian with many small terms.
 *
 * Part 2 compares all three methods inside textbook quantum phase estimation, including
 * qubitisation, which block encodes the Hamiltonian rather than approximating time evolution.
 *
 * The results of each part, and where each method performs best, are discussed in main().
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <quest.h>

#include "quiche/mappings.hpp"
#include "quiche/qdrift.hpp"
#include "quiche/qpe.hpp"
#include "quiche/quest-patches.hpp"

namespace {

// Number of Pauli rotations in one Trotterised evolution:
//   order 1     : numTerms * reps
//   order 2k>=2 : 2 * 5^(k-1) * numTerms * reps
long getTrotterCost(int order, int reps, qindex numTerms) {
    if (order == 1) {
        return numTerms * reps;
    }

    // Each increase of the order by 2 nests five copies of the lower-order formula
    long factor = 2;
    for (int k = 2; k < order; k += 2) {
        factor *= 5;
    }

    return factor * numTerms * reps;
}

// Random Hamiltonian with a few large and many small terms, where QDRIFT should do well
PauliStrSum getSkewedRandomPauliStrSum(int numTerms, int numQubits, std::mt19937_64 &rng) {
    std::uniform_int_distribution<> dist(0, 3);
    std::vector<PauliStr> strings(numTerms);
    std::vector<qcomp> coeffs(numTerms);

    for (int i = 0; i < numTerms; i++) {
        std::string string;
        for (int j = 0; j < numQubits; j++) {
            string.push_back("IXYZ"[dist(rng)]);
        }
        strings[i] = getPauliStr(string);

        // Exponentially decaying weights keep lambda small while the number of terms is large
        coeffs[i] = qcomp(std::exp(-0.25 * i), 0.0);
    }
    return createPauliStrSum(strings, coeffs);
}

void compareTimeEvolution(const std::string &name, PauliStrSum hamiltonian, int numQubits, double time,
                          std::mt19937_64 &rng) {

    std::cout << "\n---------- " << name << " (" << hamiltonian.numTerms
              << " terms, lambda = " << qdrift::getPauliStrSumNorm(hamiltonian) << ", t = " << time << ") ----------\n"
              << "method   order  reps    rotations  fidelity\n";

    // A high-order, many-step Trotterisation serves as the exact evolution
    Qureg reference = createQureg(numQubits);
    initPlusState(reference);
    applyTrotterizedUnitaryTimeEvolution(reference, hamiltonian, time, 4, 200);

    // Trotterisation is deterministic, so a single run per setting suffices
    std::cout << std::setprecision(6);
    Qureg trialQureg = createQureg(numQubits);
    for (int order : {1, 2, 4}) {
        for (int reps : {1, 4, 16}) {
            initPlusState(trialQureg);
            applyTrotterizedUnitaryTimeEvolution(trialQureg, hamiltonian, time, order, reps);
            qreal fidelity = calcFidelity(reference, trialQureg);
            long cost = getTrotterCost(order, reps, hamiltonian.numTerms);
            std::cout << "Trotter" << "  " << order << "\t" << reps << "\t" << cost << "\t   " << fidelity << "\n";
        }
    }

    // QDRIFT applies `reps` randomly sampled rotations, independent of the number of terms.
    // Its accuracy guarantee holds for the average over runs, so the fidelity is averaged.
    int numTrials = 20;
    for (int reps : {10, 100, 1000, 10000}) {
        qreal averageFidelity = 0;
        for (int i = 0; i < numTrials; i++) {
            initPlusState(trialQureg);
            qdrift::applyQDRIFTUnitaryTimeEvolution(trialQureg, hamiltonian, time, reps, rng);
            qreal fidelity = calcFidelity(reference, trialQureg);
            averageFidelity += fidelity / numTrials;
        }
        std::cout << "QDRIFT" << "  " << " -" << "\t" << reps << "\t" << reps << "\t   " << averageFidelity << "\n";
    }

    destroyQureg(trialQureg);
    destroyQureg(reference);
}

void compareQPE(PauliStrSum hamiltonian, qreal idCoeff, qindex initialState, int numDataQubits, int numQPEAncillas,
                double exactEnergy, std::mt19937_64 &rng) {

    // Textbook QPE applies controlled-U^(2^k) for k = 0..n-1, i.e. U is applied 2^n - 1 times
    // in total. The cost of one application of U is:
    //   Trotter   : getTrotterCost(order, reps, numTerms) Pauli rotations
    //   QDRIFT    : reps Pauli rotations
    //   Qubitised : one walk step, i.e. one SELECT over all numTerms Paulis plus PREP and PREP†
    long numApplications = (1 << numQPEAncillas) - 1;

    double t = 1.0;
    int order = 2;
    int trotterReps = 10;
    int qdriftReps = 100;

    std::cout << "\n---------- Textbook QPE with " << numQPEAncillas << " ancillas ----------\n"
              << "method       qubits  cost          energy      |err|\n";

    auto printRow = [exactEnergy](const std::string &method, int numQubits, const std::string &cost, double energy) {
        std::cout << std::left << std::setw(13) << method << std::setw(8) << numQubits << std::setw(14) << cost
                  << std::setw(12) << energy << std::abs(energy - exactEnergy) << '\n';
    };

    // Data register: [0, numDataQubits)
    // QPE register: [numDataQubits, numDataQubits + numQPEAncillas)
    // Qubitisation register: after the QPE register, indexing the terms of the Hamiltonian
    int numQubitisationAncillas = static_cast<int>(std::ceil(std::log2(hamiltonian.numTerms)));
    std::vector<int> qpeAncillas(numQPEAncillas);
    std::iota(qpeAncillas.begin(), qpeAncillas.end(), numDataQubits);

    std::vector<int> qubitisationAncillas(numQubitisationAncillas);
    std::iota(qubitisationAncillas.begin(), qubitisationAncillas.end(), numDataQubits + numQPEAncillas);

    // Trotterisation approximates U = exp(iHt), so the phase is E t / 2 pi
    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    initClassicalState(qureg, initialState);
    qreal phase = qpe::getPhaseTextbookTrotter(qureg, hamiltonian, qpeAncillas, order, trotterReps, t);
    qreal energy = qpe::getEnergyFromTrotterPhase(phase, t, idCoeff);
    long trotterCost = numApplications * getTrotterCost(order, trotterReps, hamiltonian.numTerms);
    printRow("Trotter", qureg.numQubits, std::to_string(trotterCost) + " rot", energy);

    // QDRIFT approximates the same U, so it shares the register layout and phase-to-energy map.
    // QPE modifies the state, so it must be prepared again before reusing the Qureg.
    initClassicalState(qureg, initialState);
    phase = qpe::getPhaseTextbookQDRIFT(qureg, hamiltonian, qpeAncillas, qdriftReps, t, rng);
    energy = qpe::getEnergyFromTrotterPhase(phase, t, idCoeff);
    printRow("QDRIFT", qureg.numQubits, std::to_string(numApplications * qdriftReps) + " rot", energy);

    // The qubitisation walk operator has eigenphases +-arccos(E / lambda) rather than E t, so
    // there is no evolution time and the energy is recovered as lambda cos(2 pi phase)
    Qureg qubitisationQureg = createQureg(numDataQubits + numQPEAncillas + numQubitisationAncillas);
    initClassicalState(qubitisationQureg, initialState);
    phase = qpe::getPhaseTextbookQubitised(qubitisationQureg, hamiltonian, qpeAncillas, qubitisationAncillas);
    energy = qpe::getEnergyFromQubitisationPhase(phase, qdrift::getPauliStrSumNorm(hamiltonian), idCoeff);
    printRow("Qubitisation", qubitisationQureg.numQubits, std::to_string(numApplications) + " walks", energy);

    destroyQureg(qureg);
    destroyQureg(qubitisationQureg);
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

    std::mt19937_64 rng(965456);

    // Part 1: time evolution
    compareTimeEvolution("H2", hamiltonian, numDataQubits, 1.0, rng);

    // QDRIFT's advantage appears for many small terms, and grows relative to a single Trotter
    // step at longer times
    PauliStrSum randomSkewHamiltonian = getSkewedRandomPauliStrSum(200, 6, rng);
    compareTimeEvolution("Random Skew", randomSkewHamiltonian, 6, 1.0, rng);
    compareTimeEvolution("Random Skew", randomSkewHamiltonian, 6, 5.0, rng);
    destroyPauliStrSum(randomSkewHamiltonian);

    // Where each method performs best for time evolution:
    // - Trotterisation suits Hamiltonians with few terms of similar size, such as H2, where a single
    //   second-order step (28 rotations) already reaches a fidelity of 0.99995. Its cost grows with
    //   the number of terms, but higher orders converge quickly once enough steps are affordable.
    // - QDRIFT suits Hamiltonians with many terms dominated by a few large ones, at low precision.
    //   Its cost scales as lambda^2 t^2 / eps independently of the number of terms, so for the
    //   200-term Hamiltonian it is the only option cheaper than one Trotter step (100 rotations give
    //   a fidelity of 0.84), and at t = 5 it reaches 0.62 with 1000 rotations where Trotterisation
    //   reaches 0.08 with 800. Its 1/N convergence means Trotterisation wins again at high precision.

    // Part 2: phase estimation
    qindex hf = mappings::getHartreeFockStateJW(numElectrons);
    compareQPE(hamiltonian, idCoeff.real(), hf, numDataQubits, 6, exactEnergy, rng);

    // With 6 QPE ancillas Trotterisation and QDRIFT give the same energy, as the error is dominated
    // by the phase resolution 2 pi / (2^6 t) rather than by the simulation method.
    // Qubitisation has no time-step error: its accuracy is limited only by the QPE phase grid.
    // It needs ceil(log2(numTerms)) extra ancillas, and its walk steps are not comparable to Pauli
    // rotations, so a fair cost comparison needs T gate counts from the resource estimation backend.

    destroyPauliStrSum(rawHamiltonian);
    destroyPauliStrSum(hamiltonian);
    finalizeQuESTEnv();
    return 0;
}
