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
 * Demo simulation of single-ancilla iterative quantum phase estimation using Trotterisation for
 * Hamiltonian simulation with a minimal basis set H2 molecule.
 *
 * @author Vasco Ferreira
 */

#include <iostream>

#include <quest.h>
#include <quest/src/core/constants.hpp>

#include "quiche/mappings.hpp"
#include "quiche/qpe.hpp"
#include "quiche/quest-patches.hpp"

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

    PauliStrSum rawHamiltonian = createInlinePauliStrSum(hydrogenMinimalJWStr);
    qcomp idCoeff = getIdentityCoeff(rawHamiltonian);
    PauliStrSum hamiltonian = cloneWithoutIdentity(rawHamiltonian);

    int numBits = 10;
    int numAncillas = 1;
    int numDataQubits = 4;

    // Data register: [0, numDataQubits)
    // Ancilla register: numDataQubits
    Qureg qureg = createQureg(numAncillas + numDataQubits);
    int ancilla = numDataQubits;

    // Data qubit state prep
    qindex hf = mappings::getHartreeFockStateJW(numElectrons);
    initClassicalState(qureg, hf);

    // Trotter parameters
    int order = 2;
    int reps = 10;
    double t = 1;

    double phase = qpe::getPhaseIterativeTrotter(qureg, hamiltonian, ancilla, order, reps, t, numBits);
    double energy = qpe::getEnergyFromTrotterPhase(phase, t, idCoeff.real());

    std::cout << "Phase: " << phase << '\n' << "Energy: " << energy << '\n';

    destroyPauliStrSum(rawHamiltonian);
    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);
    finalizeQuESTEnv();
    return 0;
}
