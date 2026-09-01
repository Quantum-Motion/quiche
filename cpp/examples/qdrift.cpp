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
 * Demo app using QDRIFT to simulate a Hamiltonian.
 *
 * @author Vasco Ferreira
 */

#include "quiche/qdrift.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int verbosity = 1;

PauliStr getRandomPauliStr(int numQubits, std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> dist(0, 3);

    std::string string;
    string.reserve(numQubits);

    for (int i = 0; i < numQubits; i++) {
        string.push_back("IXYZ"[dist(rng)]);
    }

    return getPauliStr(string);
}

PauliStrSum getRandomRealPauliStrSum(int numTerms, int numQubits, std::mt19937_64 &rng) {
    // arbitrary range, no particular reason for (0, 1)
    std::uniform_real_distribution<qreal> dist(0, 1);

    std::vector<PauliStr> strings(numTerms);
    std::vector<qcomp> coeffs(numTerms);

    for (int i = 0; i < numTerms; i++) {
        coeffs[i] = qcomp(dist(rng), 0.0);
        strings[i] = getRandomPauliStr(numQubits, rng);
    }

    return createPauliStrSum(strings, coeffs);
}

int main() {
    initQuESTEnv();

    int numTerms = 50;
    int numQubits = 5;
    double time = 1;
    double eps = 0.001;

    std::random_device rd;
    std::mt19937_64 rng(rd());
    PauliStrSum hamiltonian = getRandomRealPauliStrSum(numTerms, numQubits, rng);

    Qureg qdrift_qureg = createQureg(numQubits);
    initPlusState(qdrift_qureg);

    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int reps = std::ceil(2 * norm * norm * time * time / eps);
    qdrift::applyQDRIFTUnitaryTimeEvolution(qdrift_qureg, hamiltonian, time, reps, rng);

    if (verbosity >= 2) {
        reportQureg(qdrift_qureg);
    }

    Qureg trotter_qureg = createQureg(numQubits);
    initPlusState(trotter_qureg);
    applyTrotterizedUnitaryTimeEvolution(trotter_qureg, hamiltonian, time, 4, 100);

    if (verbosity >= 2) {
        reportQureg(trotter_qureg);
    }

    std::cout << "Fidelity = " << calcFidelity(qdrift_qureg, trotter_qureg) << '\n';

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qdrift_qureg);
    destroyQureg(trotter_qureg);
    finalizeQuESTEnv();
    return 0;
}
