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

#include "quiche/qpe.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <quest.h>
#include <quest/src/core/constants.hpp>
#include <quest/src/core/paulilogic.hpp>

#include "quiche/qdrift.hpp"

TEST_CASE("Textbook Trotter QPE", "[qpe][textbook][trotter]") {

    int numQPEAncillas = 8;
    int order = 2;
    int reps = 10;

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;

    std::vector<int> qpeAncillas(numQPEAncillas);
    std::iota(qpeAncillas.begin(), qpeAncillas.end(), numDataQubits);

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    stateprepLambda(qureg);

    double t = 2 * const_PI / (1.25 * (2 * norm));
    double phase = qpe::getPhaseTextbookTrotter(qureg, hamiltonian, qpeAncillas, order, reps, t);
    double actualEnergy = phase * (2 * const_PI / t);

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Textbook QDRIFT QPE", "[qpe][textbook][qdrift]") {
    int numQPEAncillas = 8;
    int reps = 100;
    std::mt19937_64 rng(126234);

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;

    std::vector<int> qpeAncillas(numQPEAncillas);
    std::iota(qpeAncillas.begin(), qpeAncillas.end(), numDataQubits);

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    stateprepLambda(qureg);

    double t = 2 * const_PI / (1.25 * (2 * norm));
    double phase = qpe::getPhaseTextbookQDRIFT(qureg, hamiltonian, qpeAncillas, reps, t, rng);
    double actualEnergy = phase * (2 * const_PI / t);

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Textbook Qubitised QPE", "[qpe][textbook][qubitised]") {
    int numQPEAncillas = 8;

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;
    int numQubitisationAncillas = std::max(std::ceil(std::log2(hamiltonian.numTerms)), 1.0);

    std::vector<int> qpeAncillas(numQPEAncillas);
    std::iota(qpeAncillas.begin(), qpeAncillas.end(), numDataQubits);

    std::vector<int> qubitisationAncillas(numQubitisationAncillas);
    std::iota(qubitisationAncillas.begin(), qubitisationAncillas.end(), numDataQubits + numQPEAncillas);

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas + numQubitisationAncillas);
    stateprepLambda(qureg);

    double phase = qpe::getPhaseTextbookQubitised(qureg, hamiltonian, qpeAncillas, qubitisationAncillas);
    double actualEnergy = std::cos(phase * 2 * const_PI) * norm;

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Textbook Qubitised Optimised QPE", "[qpe][textbook][qubitised-optimised]") {
    int numQPEAncillas = 8;

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;
    int numQubitisationAncillas = std::max(std::ceil(std::log2(hamiltonian.numTerms)), 1.0);

    std::vector<int> qpeAncillas(numQPEAncillas);
    std::iota(qpeAncillas.begin(), qpeAncillas.end(), numDataQubits);

    std::vector<int> qubitisationAncillas(numQubitisationAncillas);
    std::iota(qubitisationAncillas.begin(), qubitisationAncillas.end(), numDataQubits + numQPEAncillas);

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas + numQubitisationAncillas);
    stateprepLambda(qureg);

    double phase = qpe::getPhaseTextbookQubitisedOptimised(qureg, hamiltonian, qpeAncillas, qubitisationAncillas);
    double actualEnergy = std::cos(phase * 2 * const_PI) * norm;

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Kitaev Trotter QPE", "[qpe][kitaev][trotter]") {
    int numQPEAncillas = 1;
    int order = 2;
    int reps = 10;
    int rounds = 8;

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;

    int ancilla = numDataQubits;

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    stateprepLambda(qureg);

    double t = 2 * const_PI / (1.25 * (2 * norm));
    double phase = qpe::getPhaseKitaevTrotter(qureg, hamiltonian, ancilla, order, reps, t, rounds);
    double actualEnergy = phase * (2 * const_PI / t);

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Kitaev QDRIFT QPE", "[qpe][kitaev][qdrift]") {
    int numQPEAncillas = 1;
    int reps = 100;
    std::mt19937_64 rng(7483);
    int rounds = 8;

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;

    int ancilla = numDataQubits;

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    stateprepLambda(qureg);

    double t = 2 * const_PI / (1.25 * (2 * norm));
    double phase = qpe::getPhaseKitaevQDRIFT(qureg, hamiltonian, ancilla, reps, t, rounds, rng);
    double actualEnergy = phase * (2 * const_PI / t);

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Naive Trotter QPE", "[qpe][naive][trotter]") {
    int numQPEAncillas = 1;
    int order = 2;
    int reps = 10;

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;

    int ancilla = numDataQubits;

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    stateprepLambda(qureg);

    double t = 2 * const_PI / (1.25 * (2 * norm));
    double phase = qpe::getPhaseNaiveTrotter(qureg, hamiltonian, ancilla, order, reps, t);
    double actualEnergy = phase * (2 * const_PI / t);

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Naive QDRIFT QPE", "[qpe][naive][qdrift]") {
    int numQPEAncillas = 1;
    int reps = 100;
    std::mt19937_64 rng(7483);

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;

    int ancilla = numDataQubits;

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    stateprepLambda(qureg);

    double t = 2 * const_PI / (1.25 * (2 * norm));
    double phase = qpe::getPhaseNaiveQDRIFT(qureg, hamiltonian, ancilla, reps, t, rng);
    double actualEnergy = phase * (2 * const_PI / t);

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Iterative Trotter QPE", "[qpe][iterative][trotter]") {
    int numQPEAncillas = 1;
    int order = 2;
    int reps = 10;
    int rounds = 8;

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;
    int ancilla = numDataQubits;

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    stateprepLambda(qureg);

    double t = 2 * const_PI / (1.25 * (2 * norm));
    double phase = qpe::getPhaseIterativeTrotter(qureg, hamiltonian, ancilla, order, reps, t, rounds);
    double actualEnergy = phase * (2 * const_PI / t);

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}

TEST_CASE("Iterative QDRIFT QPE", "[qpe][iterative][qdrift]") {
    int numQPEAncillas = 1;
    int reps = 100;
    std::mt19937_64 rng(7483);
    int rounds = 8;

    auto [hamiltonianString, expectedEnergy, stateprepLambda] = GENERATE(
        table<std::string, double, void (*)(Qureg)>({{"1 Z", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 ZZ", -1.0, [](Qureg qureg) { initClassicalState(qureg, 1); }},
                                                     {"1 XX \n 1 YY \n 1 ZZ", -3.0, [](Qureg qureg) {
                                                          applyHadamard(qureg, 0);
                                                          applyControlledPauliX(qureg, 0, 1);
                                                          applyPauliX(qureg, 1);
                                                          applyPauliZ(qureg, 1);
                                                      }}}));

    PauliStrSum hamiltonian = createInlinePauliStrSum(hamiltonianString);
    qreal norm = qdrift::getPauliStrSumNorm(hamiltonian);
    int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(hamiltonian) + 1;

    int ancilla = numDataQubits;

    Qureg qureg = createQureg(numDataQubits + numQPEAncillas);
    stateprepLambda(qureg);

    double t = 2 * const_PI / (1.25 * (2 * norm));
    double phase = qpe::getPhaseIterativeQDRIFT(qureg, hamiltonian, ancilla, reps, t, rounds, rng);
    double actualEnergy = phase * (2 * const_PI / t);

    destroyPauliStrSum(hamiltonian);
    destroyQureg(qureg);

    REQUIRE(actualEnergy == Catch::Approx(expectedEnergy).epsilon(0.01));
}
