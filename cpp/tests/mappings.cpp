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

#include "quiche/mappings.hpp"

#include <bitset>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_range.hpp>

namespace Gen = Catch::Generators;

int numSamples = 20;
int numQubits = 10;

TEST_CASE("getHartreeFockStateJW", "[mappings][jordan_wigner][hartree_fock]") {
    int numElectrons = GENERATE(Gen::take(numSamples, Gen::random(0, numQubits)));

    std::bitset<64> bits;
    for (int i = 0; i < numElectrons; i++)
        bits.set(i);

    qindex expected = bits.to_ullong();
    qindex actual = mappings::getHartreeFockStateJW(numElectrons);

    REQUIRE(actual == expected);
}

TEST_CASE("getHartreeFockStateBK", "[mappings][bravyi_kitaev][hartree_fock]") {
    int numElectrons = GENERATE(Gen::take(numSamples, Gen::random(0, numQubits)));

    // Initalise bitstring to occupation
    std::bitset<64> bits;
    for (int i = 0; i < numElectrons; i++)
        bits.set(i);

    for (int i = 0; i < numQubits; i++) {
        int parent = i | (i + 1);
        if (parent < numQubits)
            bits[parent] = bits[parent] ^ bits[i];
    }

    qindex expected = bits.to_ullong();
    qindex actual = mappings::getHartreeFockStateBK(numElectrons, numQubits);

    REQUIRE(actual == expected);
}

TEST_CASE("getHartreeFockStateParity", "[mappings][parity][hartree_fock]") {
    int numElectrons = GENERATE(Gen::take(numSamples, Gen::random(0, numQubits)));

    std::bitset<64> bits;
    int parity = 0;

    // naive prefix sum calculation
    for (int i = 0; i < numQubits; i++) {
        parity ^= (i < numElectrons);
        bits[i] = parity;
    }

    qindex expected = bits.to_ullong();
    qindex actual = mappings::getHartreeFockStateParity(numElectrons, numQubits);

    REQUIRE(actual == expected);
}

TEST_CASE("getQubitBasisStateJW", "[mappings][jordan_wigner]") {

    auto occupation = GENERATE(from_range(
        std::vector<std::vector<int>>{{1, 1, 1, 1, 0, 0, 0, 0}, {0, 1, 1, 0, 1, 0, 1}, {0, 0, 1, 1, 0}, {1, 0, 1, 0}}));

    // Jordan-Wigner qubit basis state should match the spin basis state
    REQUIRE(mappings::getQubitBasisStateJW(occupation) == occupation);
}

TEST_CASE("getQubitBasisStateParity", "[mappings][parity]") {

    auto [occupation, expected] =
        GENERATE(table<std::vector<int>, std::vector<int>>({{{1, 1, 1, 1, 0, 0, 0, 0}, {1, 0, 1, 0, 0, 0, 0, 0}},
                                                            {{0, 1, 1, 0, 1, 0, 1}, {0, 1, 0, 0, 1, 1, 0}},
                                                            {{0, 0, 1, 1, 0}, {0, 0, 1, 0, 0}},
                                                            {{1, 0, 1, 0}, {1, 1, 0, 0}}}));

    REQUIRE(mappings::getQubitBasisStateParity(occupation) == expected);
}

TEST_CASE("getQubitBasisStateBK", "[mappings][bravyi_kitaev]") {

    auto [occupation, expected] =
        GENERATE(table<std::vector<int>, std::vector<int>>({{{1, 1, 1, 1, 0, 0, 0, 0}, {1, 0, 1, 0, 0, 0, 0, 0}},
                                                            {{0, 1, 1, 0, 1, 0, 1}, {0, 1, 1, 0, 1, 1, 1}},
                                                            {{0, 0, 1, 1, 0}, {0, 0, 1, 0, 0}},
                                                            {{1, 0, 1, 0}, {1, 1, 1, 0}}}));

    REQUIRE(mappings::getQubitBasisStateBK(occupation) == expected);
}
