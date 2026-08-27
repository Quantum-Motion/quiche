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
 * Demo simulation of textbook quantum phase estimation using qubitisation
 * for Hamiltonian simulation with a minimal basis set H2 molecule.
 *
 * @author Vasco Ferreira
 */

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include <quest.h>
#include <quest/src/core/constants.hpp>

#include "quiche/mappings.hpp"
#include "quiche/qdrift.hpp"
#include "quiche/qpe.hpp"
#include "quiche/quest-patches.hpp"
#include "quiche/utils.hpp"

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

    int numQPEAncillas = 8;
    int numDataQubits = 4;
    int numQubitisationAncillas = std::ceil(std::log2(hamiltonian.numTerms));

    int numAncillas = numQPEAncillas + numQubitisationAncillas;
    int numQubits = numDataQubits + numAncillas;

    std::vector<int> dataQubits(numDataQubits);
    std::iota(dataQubits.begin(), dataQubits.end(), 0);

    std::vector<int> qpeAncillas(numQPEAncillas);
    std::iota(qpeAncillas.begin(), qpeAncillas.end(), numDataQubits);

    std::vector<int> qubitisationAncillas(numQubitisationAncillas);
    std::iota(qubitisationAncillas.begin(), qubitisationAncillas.end(), numDataQubits + numQPEAncillas);

    std::cout << "---------- Registers ----------\n"
              << "Data: " << formatVector(dataQubits) << "\n"
              << "QPE: " << formatVector(qpeAncillas) << "\n"
              << "Qubitisation: " << formatVector(qubitisationAncillas) << "\n"
              << "-------------------------------\n";

    Qureg qureg = createQureg(numQubits);

    // Data qubit state prep
    qindex hf = mappings::getHartreeFockStateJW(numElectrons);
    initClassicalState(qureg, hf);

    double phase = qpe::getPhaseTextbookQubitised(qureg, hamiltonian, qpeAncillas, qubitisationAncillas);
    auto lambda = qdrift::getPauliStrSumNorm(hamiltonian);
    double energy = qpe::getEnergyFromQubitisationPhase(phase, lambda, idCoeff.real());

    std::cout << "Phase: " << phase << '\n' << "Energy: " << energy << '\n';

    destroyPauliStrSum(rawHamiltonian);
    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);
    finalizeQuESTEnv();
    return 0;
}
