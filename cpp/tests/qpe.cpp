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
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <quest.h>
#include <quest/src/core/constants.hpp>
#include <quest/src/core/paulilogic.hpp>

#include "quiche/qdrift.hpp"
#include "quiche/quest-patches.hpp"

namespace {

struct QPESystem {
    std::string name;
    std::string hamiltonian;
    std::function<void(Qureg)> prepareState;
    double expectedEnergy;
    double margin;
};

struct QPEContext {
    Qureg qureg;
    PauliStrSum hamiltonian;
    qreal norm;
    qreal t;
    qreal identityCoeff;
    std::vector<int> qpeAncillas;
    std::vector<int> qubitisationAncillas;
};

template <typename EstimationFn>
void checkQPE(const QPESystem &system, EstimationFn estimate, int numQPEAncillas, bool allocateQubitisationAncillas) {

    SECTION("Hamiltonian: " + system.name) {
        PauliStrSum raw = createInlinePauliStrSum(system.hamiltonian);

        QPEContext ctx{};
        ctx.identityCoeff = getIdentityCoeff(raw).real();
        ctx.hamiltonian = cloneWithoutIdentity(raw);
        destroyPauliStrSum(raw);

        ctx.norm = qdrift::getPauliStrSumNorm(ctx.hamiltonian);
        ctx.t = 2 * const_PI / (1.25 * (2 * ctx.norm)); // time with some headroom to avoid aliasing

        // Register layout: [data qubits | QPE ancillas | qubitisation ancillas]
        int numDataQubits = paulis_getIndOfLefmostNonIdentityPauli(ctx.hamiltonian) + 1;
        int numQubitisationAncillas =
            allocateQubitisationAncillas
                ? static_cast<int>(std::max(std::ceil(std::log2(ctx.hamiltonian.numTerms)), 1.0))
                : 0;

        ctx.qureg = createQureg(numDataQubits + numQPEAncillas + numQubitisationAncillas);
        system.prepareState(ctx.qureg);

        ctx.qpeAncillas = std::vector<int>(numQPEAncillas);
        std::iota(ctx.qpeAncillas.begin(), ctx.qpeAncillas.end(), numDataQubits);

        ctx.qubitisationAncillas = std::vector<int>(numQubitisationAncillas);
        std::iota(ctx.qubitisationAncillas.begin(), ctx.qubitisationAncillas.end(), numDataQubits + numQPEAncillas);

        double energy = estimate(ctx);

        destroyQureg(ctx.qureg);
        destroyPauliStrSum(ctx.hamiltonian);

        REQUIRE(energy == Catch::Approx(system.expectedEnergy).margin(system.margin));
    }
}

const std::string hydrogenMinimalJW = R"(
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

// Systems with exact eigenstates only (compatible with every method)
const std::vector<QPESystem> exactEigenstateSystems = {
    {"Z", "1 Z", [](Qureg q) { initClassicalState(q, 1); }, -1.0, 0.02},
    {"ZZ", "1 ZZ", [](Qureg q) { initClassicalState(q, 1); }, -1.0, 0.02},
    {"XX+YY+ZZ", "1 XX \n 1 YY \n 1 ZZ",
     [](Qureg q) {
         applyHadamard(q, 0);
         applyControlledPauliX(q, 0, 1);
         applyPauliX(q, 1);
         applyPauliZ(q, 1);
     },
     -3.0, 0.05},
};

// All systems including H2 with approximate HF eigenstate
const std::vector<QPESystem> allSystems = [] {
    std::vector<QPESystem> all = exactEigenstateSystems;
    all.push_back({"H2", hydrogenMinimalJW, [](Qureg q) { initClassicalState(q, 3); }, -1.137, 0.02});
    return all;
}();

} // namespace

TEST_CASE("Textbook Trotter QPE", "[qpe][textbook][trotter]") {
    int numQPEAncillas = 8;
    int order = 2;
    int reps = 5;
    bool allocateQubitisationAncillas = false;

    auto system = GENERATE(from_range(allSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase = qpe::getPhaseTextbookTrotter(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas, order, reps, ctx.t);
        return qpe::getEnergyFromTrotterPhase(phase, ctx.t, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Textbook QDRIFT QPE", "[qpe][textbook][qdrift]") {
    int numQPEAncillas = 8;
    int reps = 1000;
    std::mt19937_64 rng(126234);
    bool allocateQubitisationAncillas = false;

    auto system = GENERATE(from_range(allSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase = qpe::getPhaseTextbookQDRIFT(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas, reps, ctx.t, rng);
        return qpe::getEnergyFromTrotterPhase(phase, ctx.t, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Textbook Qubitised QPE", "[qpe][textbook][qubitised]") {
    int numQPEAncillas = 8;
    bool allocateQubitisationAncillas = true;

    auto system = GENERATE(from_range(allSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase =
            qpe::getPhaseTextbookQubitised(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas, ctx.qubitisationAncillas);
        return qpe::getEnergyFromQubitisationPhase(phase, ctx.norm, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Textbook Qubitised Optimised QPE", "[qpe][textbook][qubitised-optimised]") {
    int numQPEAncillas = 8;
    bool allocateQubitisationAncillas = true;

    auto system = GENERATE(from_range(allSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase = qpe::getPhaseTextbookQubitisedOptimised(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas,
                                                               ctx.qubitisationAncillas);
        return qpe::getEnergyFromQubitisationPhase(phase, ctx.norm, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Kitaev Trotter QPE", "[qpe][kitaev][trotter]") {
    int order = 2;
    int reps = 5;
    int rounds = 8;
    int numQPEAncillas = 1;
    bool allocateQubitisationAncillas = false;

    auto system = GENERATE(from_range(allSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase =
            qpe::getPhaseKitaevTrotter(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas[0], order, reps, ctx.t, rounds);
        return qpe::getEnergyFromTrotterPhase(phase, ctx.t, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Kitaev QDRIFT QPE", "[qpe][kitaev][qdrift]") {
    int reps = 1000;
    int rounds = 8;
    std::mt19937_64 rng(7483);
    int numQPEAncillas = 1;
    bool allocateQubitisationAncillas = false;

    auto system = GENERATE(from_range(allSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase =
            qpe::getPhaseKitaevQDRIFT(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas[0], reps, ctx.t, rounds, rng);
        return qpe::getEnergyFromTrotterPhase(phase, ctx.t, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Naive Trotter QPE", "[qpe][naive][trotter]") {
    int order = 2;
    int reps = 5;
    int numQPEAncillas = 1;
    bool allocateQubitisationAncillas = false;

    // Naive QPE yields t-dependent weighted average of eigenenergies so not tested with approximate HF eigenstate
    auto system = GENERATE(from_range(exactEigenstateSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase = qpe::getPhaseNaiveTrotter(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas[0], order, reps, ctx.t);
        return qpe::getEnergyFromTrotterPhase(phase, ctx.t, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Naive QDRIFT QPE", "[qpe][naive][qdrift]") {
    int reps = 1000;
    std::mt19937_64 rng(7483);
    int numQPEAncillas = 1;
    bool allocateQubitisationAncillas = false;

    // Naive QPE yields t-dependent weighted average of eigenenergies so not tested with approximate HF eigenstate
    auto system = GENERATE(from_range(exactEigenstateSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase = qpe::getPhaseNaiveQDRIFT(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas[0], reps, ctx.t, rng);
        return qpe::getEnergyFromTrotterPhase(phase, ctx.t, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Iterative Trotter QPE", "[qpe][iterative][trotter]") {
    int order = 2;
    int reps = 5;
    int rounds = 8;
    int numQPEAncillas = 1;
    bool allocateQubitisationAncillas = false;

    auto system = GENERATE(from_range(allSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase =
            qpe::getPhaseIterativeTrotter(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas[0], order, reps, ctx.t, rounds);
        return qpe::getEnergyFromTrotterPhase(phase, ctx.t, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}

TEST_CASE("Iterative QDRIFT QPE", "[qpe][iterative][qdrift]") {
    int reps = 1000;
    int rounds = 8;
    std::mt19937_64 rng(7483);
    int numQPEAncillas = 1;
    bool allocateQubitisationAncillas = false;

    auto system = GENERATE(from_range(allSystems));

    auto estimateEnergy = [&](const QPEContext &ctx) {
        double phase =
            qpe::getPhaseIterativeQDRIFT(ctx.qureg, ctx.hamiltonian, ctx.qpeAncillas[0], reps, ctx.t, rounds, rng);
        return qpe::getEnergyFromTrotterPhase(phase, ctx.t, ctx.identityCoeff);
    };

    checkQPE(system, estimateEnergy, numQPEAncillas, allocateQubitisationAncillas);
}
