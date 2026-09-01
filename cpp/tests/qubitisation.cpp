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

#include "quiche/qubitisation.hpp"

#include <bitset>
#include <cmath>
#include <complex>
#include <cstddef>
#include <tuple>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <quest.h>

namespace {
std::tuple<std::vector<qreal>, std::vector<qreal>> getQuregAmpsRealImagParts(Qureg qureg, qindex startInd,
                                                                             qindex numAmps) {

    std::vector<qcomp> combined = getQuregAmps(qureg, startInd, numAmps);
    std::vector<qreal> real;
    std::vector<qreal> imag;

    real.reserve(combined.size());
    imag.reserve(combined.size());

    for (const auto &c : combined) {
        real.push_back(c.real());
        imag.push_back(c.imag());
    }

    return {real, imag};
}
} // namespace

TEST_CASE("applySelect", "[qubitisation][select]") {

    SECTION("Real positive coefficients") {

        // Specific value of coefficients is irrelevant
        std::vector<qcomp> coeffs = {1, 1, 1, 1};
        std::vector<PauliStr> paulis{getPauliStr("I"), getPauliStr("X"), getPauliStr("Y"), getPauliStr("Z")};
        PauliStrSum sum = createPauliStrSum(paulis, coeffs);

        Qureg actual = createQureg(3);
        Qureg expected = createQureg(3);
        initRandomPureState(actual);
        setQuregToClone(expected, actual);

        qubitisation::applySelect(actual, sum, {1, 2});

        for (int i = 0; i < 4; i++) {

            // Quick and dirty integer to binary (in vector<int> format) conversion
            std::bitset<2> b(i);
            std::vector<int> states(b.size());
            for (std::size_t i = 0; i < b.size(); i++) {
                states[i] = b[i];
            }

            applyMultiStateControlledPauliStr(expected, {1, 2}, states, sum.strings[i]);
        }

        auto [actualReal, actualImag] = getQuregAmpsRealImagParts(actual, 0, 8);
        auto [expectedReal, expectedImag] = getQuregAmpsRealImagParts(expected, 0, 8);

        REQUIRE_THAT(actualReal, Catch::Matchers::Approx(expectedReal).margin(1e-15));
        REQUIRE_THAT(actualImag, Catch::Matchers::Approx(expectedImag).margin(1e-15));

        destroyPauliStrSum(sum);
        destroyQureg(actual);
        destroyQureg(expected);
    }

    SECTION("General coefficients") {
        std::vector<qcomp> coeffs = {-2.5, 1};
        std::vector<PauliStr> paulis{getPauliStr("X"), getPauliStr("Z")};
        PauliStrSum sum = createPauliStrSum(paulis, coeffs);

        Qureg actual = createQureg(2);
        Qureg expected = createQureg(2);

        qubitisation::applySelect(actual, sum, {1});

        // Control qubit (qubit 1) is in 0 state so only the first term is applied
        applyPauliStr(expected, sum.strings[0]);
        applyPhaseShift(expected, 0, std::arg(sum.coeffs[0]));

        auto [actualReal, actualImag] = getQuregAmpsRealImagParts(actual, 0, 4);
        auto [expectedReal, expectedImag] = getQuregAmpsRealImagParts(expected, 0, 4);

        REQUIRE_THAT(actualReal, Catch::Matchers::Approx(expectedReal).margin(1e-15));
        REQUIRE_THAT(actualImag, Catch::Matchers::Approx(expectedImag).margin(1e-15));

        destroyPauliStrSum(sum);
        destroyQureg(actual);
        destroyQureg(expected);
    }
}

TEST_CASE("applyPrep", "[qubitisation][prep]") {

    std::vector<qcomp> coeffs = {1.0, -0.25, -2, 0.5, 1};

    // Paulis are irrelevant for this test
    PauliStr id = getPauliStr("I");
    std::vector<PauliStr> paulis{id, id, id, id, id};
    PauliStrSum sum = createPauliStrSum(paulis, coeffs);

    std::vector<qreal> magnitudes = {1.0, 0.25, 2, 0.5, 1};
    qreal lambda = 1.0 + 0.25 + 2 + 0.5 + 1;

    std::vector<int> qubits = {0, 1, 2};

    SECTION("Forward") {

        std::vector<qreal> expectedReal(8, 0);
        std::vector<qreal> expectedImag(8, 0);
        for (int i = 0; i < coeffs.size(); i++) {
            expectedReal[i] = std::sqrt(magnitudes[i] / lambda);
        }

        Qureg qureg = createQureg(3);
        qubitisation::applyPauliStrSumPrep(qureg, sum, qubits, false);

        auto [actualReal, actualImag] = getQuregAmpsRealImagParts(qureg, 0, 8);
        REQUIRE_THAT(actualReal, Catch::Matchers::Approx(expectedReal).margin(1e-15));
        REQUIRE_THAT(actualImag, Catch::Matchers::Approx(expectedImag).margin(1e-15));

        destroyPauliStrSum(sum);
        destroyQureg(qureg);
    }

    SECTION("Forward and Inverse") {

        std::vector<qreal> expectedReal(8, 0);
        expectedReal[0] = 1.0;
        std::vector<qreal> expectedImag(8, 0);

        Qureg qureg = createQureg(3);
        PauliStrSum sum = createPauliStrSum(paulis, coeffs);

        qubitisation::applyPauliStrSumPrep(qureg, sum, qubits, false);
        qubitisation::applyPauliStrSumPrep(qureg, sum, qubits, true);

        auto [actualReal, actualImag] = getQuregAmpsRealImagParts(qureg, 0, 8);

        REQUIRE_THAT(actualReal, Catch::Matchers::Approx(expectedReal).margin(1e-15));
        REQUIRE_THAT(actualImag, Catch::Matchers::Approx(expectedImag).margin(1e-15));

        destroyPauliStrSum(sum);
        destroyQureg(qureg);
    }
}

TEST_CASE("applyReflection", "[qubitisation][reflection]") {

    Qureg qureg = createQureg(3);
    initRandomPureState(qureg);

    std::vector<int> reflectionQubits = {0, 1};

    auto [expectedReal, expectedImag] = getQuregAmpsRealImagParts(qureg, 0, 8);

    // |00> ⊗ |x> states should be unchanged, all others should be flipped
    for (int i = 0; i < 8; i++) {
        if (i != 0 && i != 4) {
            expectedReal[i] *= -1;
            expectedImag[i] *= -1;
        }
    }

    qubitisation::applyReflection(qureg, reflectionQubits);
    auto [actualReal, actualImag] = getQuregAmpsRealImagParts(qureg, 0, 8);

    REQUIRE_THAT(actualReal, Catch::Matchers::Approx(expectedReal).margin(1e-15));
    REQUIRE_THAT(actualImag, Catch::Matchers::Approx(expectedImag).margin(1e-15));

    destroyQureg(qureg);
}
