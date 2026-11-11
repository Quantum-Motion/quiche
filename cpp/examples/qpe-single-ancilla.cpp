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
 * Demo comparing the single-ancilla quantum phase estimation methods (naive, iterative and
 * Kitaev) for a minimal basis set H2 molecule.
 *
 * Part 1 compares the precision of each method as the number of phase bits grows.
 * Part 2 shows how the evolution time limits naive phase estimation.
 * Part 3 repeats the comparison with QDRIFT over several random seeds.
 *
 * Note that the simulator reads ancilla expectation values exactly, whereas hardware would
 * estimate them from repeated measurements.
 *
 * The results of each part, and where each method performs best, are discussed in main().
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <quest.h>

#include "quiche/mappings.hpp"
#include "quiche/qpe.hpp"
#include "quiche/quest-patches.hpp"

namespace {

// qreal rather than double, so these also work when QuEST is compiled with single or quad precision
qreal getMean(const std::vector<qreal> &values) {
    return std::accumulate(values.begin(), values.end(), qreal(0)) / values.size();
}

// Sample standard deviation, sqrt(sum((x - mean)^2) / (n - 1))
qreal getStdDev(const std::vector<qreal> &values) {
    qreal mean = getMean(values);
    qreal sqSum = std::accumulate(values.begin(), values.end(), qreal(0),
                                  [mean](qreal accumulator, qreal x) { return accumulator + (x - mean) * (x - mean); });
    return std::sqrt(sqSum / (values.size() - 1));
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
    double exactEnergy = -1.137272; // exact diagonalisation

    PauliStrSum rawHamiltonian = createInlinePauliStrSum(hydrogenMinimalJWStr);
    qcomp idCoeff = getIdentityCoeff(rawHamiltonian);
    PauliStrSum hamiltonian = cloneWithoutIdentity(rawHamiltonian);

    int numAncillas = 1;
    int numDataQubits = 4;

    // Data register: [0, numDataQubits)
    // Ancilla register: numDataQubits
    Qureg qureg = createQureg(numAncillas + numDataQubits);
    int ancilla = numDataQubits;

    qindex hf = mappings::getHartreeFockStateJW(numElectrons);

    // Trotter parameters
    double t = 1;
    int order = 2;
    int reps = 10;

    std::cout << std::fixed << std::setprecision(6);

    // Part 1: precision vs number of bits
    std::cout << "---------- Trotter: precision vs number of bits ----------\n";

    // Naive QPE applies the controlled evolution once and has no number of bits.
    // Every QPE call modifies the state, so the HF state is prepared before each call.
    initClassicalState(qureg, hf);
    qreal phase = qpe::getPhaseNaiveTrotter(qureg, hamiltonian, ancilla, order, reps, t);
    qreal energy = qpe::getEnergyFromTrotterPhase(phase, t, idCoeff.real());
    std::cout << "method         phase           energy          |err|\n";
    std::cout << "Naive QPE      " << phase << "\t" << energy << "\t" << std::abs(energy - exactEnergy) << "\n\n\n";

    std::cout << "bits    iterative       |err|           kitaev          |err|\n";

    // Both methods determine the phase over numBits rounds, the k-th using U^(2^k)
    for (int numBits : {2, 4, 6, 8, 10, 12}) {
        initClassicalState(qureg, hf);
        qreal phaseIT = qpe::getPhaseIterativeTrotter(qureg, hamiltonian, ancilla, order, reps, t, numBits);
        qreal energyIT = qpe::getEnergyFromTrotterPhase(phaseIT, t, idCoeff.real());

        initClassicalState(qureg, hf);
        qreal phaseKT = qpe::getPhaseKitaevTrotter(qureg, hamiltonian, ancilla, order, reps, t, numBits);
        qreal energyKT = qpe::getEnergyFromTrotterPhase(phaseKT, t, idCoeff.real());

        std::cout << numBits << "\t" << energyIT << "\t" << std::abs(energyIT - exactEnergy) << "\t" << energyKT << "\t"
                  << std::abs(energyKT - exactEnergy) << "\n";
    }

    // Where each method performs best:
    // - Naive QPE needs a single controlled evolution, so it has the shortest circuit, but all of its
    //   precision comes from estimating <X> and <Y> on the ancilla. Its error is 0.013 here, and on
    //   hardware the shot noise in that estimate maps to an energy error of 2 pi dphi / t.
    // - Iterative QPE determines one bit per round from a single measurement-like decision, so it
    //   only needs few shots per round. Its error halves with each extra bit (0.53 at 2 bits,
    //   0.0001 at 12 bits), at the cost of evolutions up to 2^(numBits - 1) times longer.
    // - Kitaev QPE is accurate from only 2 bits here (error 0.0006) because the simulator reads <X>
    //   and <Y> exactly. On hardware each round needs enough shots to resolve them, but the later
    //   rounds only need to choose between two candidate phases, which tolerates some noise.

    // Part 2: naive QPE vs evolution time
    std::cout << "\n---------- Trotter: naive QPE vs evolution time ----------\n"
              << "t" << "\t\t" << "phase" << "\t\t" << "energy" << "\t\t" << "|err|" << "\n";

    // The phase E t / 2 pi wraps around once it leaves [-0.5, 0.5], after which the energy is wrong
    for (double time : {0.5, 1., 2., 3., 4., 8.}) {
        initClassicalState(qureg, hf);
        qreal timePhase = qpe::getPhaseNaiveTrotter(qureg, hamiltonian, ancilla, order, reps, time);
        qreal timeEnergy = qpe::getEnergyFromTrotterPhase(timePhase, time, idCoeff.real());
        std::cout << time << "\t" << timePhase << "\t" << timeEnergy << "\t" << std::abs(timeEnergy - exactEnergy)
                  << "\n";
    }

    // Without the identity coefficient E = -1.038, so naive QPE wraps around for
    // t > pi / 1.038 = 3.03: at t = 4 it reports an energy of +0.43. Below that limit a larger t is
    // preferable on hardware, since it shrinks the energy error caused by a given phase error.

    // Part 3: QDRIFT over several seeds
    std::cout << "\n---------- QDRIFT: 10 seeds, 8 bits ----------\n"
              << "method" << "\t\t" << "mean energy" << "\t" << "std dev" << "\t\t" << "|mean err|" << "\n";

    int qdriftReps = 200;
    int numBits = 8;
    int numSeeds = 10;

    // Each seed gives a different random compilation, so the spread over seeds shows how
    // sensitive each method is to QDRIFT's randomness
    std::vector<qreal> energyNaiveVec(numSeeds);
    std::vector<qreal> energyIterativeVec(numSeeds);
    std::vector<qreal> energyKitaevVec(numSeeds);

    for (int seed = 0; seed < numSeeds; seed++) {
        std::mt19937_64 rng(seed);

        initClassicalState(qureg, hf);
        qreal phaseNaive = qpe::getPhaseNaiveQDRIFT(qureg, hamiltonian, ancilla, qdriftReps, t, rng);
        energyNaiveVec[seed] = qpe::getEnergyFromTrotterPhase(phaseNaive, t, idCoeff.real());

        initClassicalState(qureg, hf);
        qreal phaseIT = qpe::getPhaseIterativeQDRIFT(qureg, hamiltonian, ancilla, qdriftReps, t, numBits, rng);
        energyIterativeVec[seed] = qpe::getEnergyFromTrotterPhase(phaseIT, t, idCoeff.real());

        initClassicalState(qureg, hf);
        qreal phaseKT = qpe::getPhaseKitaevQDRIFT(qureg, hamiltonian, ancilla, qdriftReps, t, numBits, rng);
        energyKitaevVec[seed] = qpe::getEnergyFromTrotterPhase(phaseKT, t, idCoeff.real());
    }

    std::cout << "Naive" << "\t\t" << getMean(energyNaiveVec) << "\t" << getStdDev(energyNaiveVec) << "\t"
              << std::abs(getMean(energyNaiveVec) - exactEnergy) << "\n";
    std::cout << "Iterative" << "\t" << getMean(energyIterativeVec) << "\t" << getStdDev(energyIterativeVec) << "\t"
              << std::abs(getMean(energyIterativeVec) - exactEnergy) << "\n";
    std::cout << "Kitaev" << "\t\t" << getMean(energyKitaevVec) << "\t" << getStdDev(energyKitaevVec) << "\t"
              << std::abs(getMean(energyKitaevVec) - exactEnergy) << "\n";

    // Naive QPE is the most sensitive to the random compilation (standard deviation 0.076 vs about
    // 0.02), since it reads the energy directly from one short random circuit, while the iterative
    // and Kitaev methods make bit-by-bit decisions that small errors do not flip.

    destroyPauliStrSum(rawHamiltonian);
    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);
    finalizeQuESTEnv();
    return 0;
}
