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
 *
 * The test data used in this file is part of the Hamlib dataset, available at
 * `https://portal.nersc.gov/cfs/m888/dcamps/hamlib/`. We do not claim copyright ownership
 * over this data. All rights remain with the respective owners.
 *
 * For more information about Hamlib see ref. [1].
 *
 * References:
 * [1] N. P. Sawaya et al., 'HamLib: A library of Hamiltonians for benchmarking quantum
 *     algorithms and hardware', Quantum, vol. 8, p. 1559, Dec. 2024, doi:
 *     10.22331/q-2024-12-11-1559.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "quiche/hamlib.hpp"

TEST_CASE("getPauliStrings", "[hamlib]") {

    SECTION("Jordan-Wigner file") {

        // Excerpt of `ham_JW-24` dataset from `NaLi.hdf5` file from Hamlib.
        // We do not claim copyright ownership over this data. All rights remain with the respective owners.
        std::string string =
            "(-152.3300792589959+0j) [] +\n"
            "(-1.1418577943835467e-05+0j) [X0 X1 Y2 Z3 Z4 Y5] +\n"
            "(-5.172629429370682e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +\n"
            "(3.20410813263569e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +\n"
            "(1.1144588463121034e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Z13 Z14 Z15 Z16 Z17 Z18 Y19] +\n"
            "(-1.1418577943835467e-05+0j) [X0 X1 X3 X4] +\n"
            "(-5.172629429370682e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +\n"
            "(3.20410813263569e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +\n"
            "(1.1144588463121034e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Z13 Z14 Z15 Z16 Z17 X18] +\n"
            "(-0.05844179549720885+0j) [X0 X1 Y4 Y5] +";

        std::vector<double> expectedCoeffs{
            -152.3300792589959,     -1.1418577943835467e-05, -5.172629429370682e-06, 3.20410813263569e-06,
            1.1144588463121034e-06, -1.1418577943835467e-05, -5.172629429370682e-06, 3.20410813263569e-06,
            1.1144588463121034e-06, -0.05844179549720885,
        };

        std::vector<std::vector<int>> expectedTargets = {
            {0},
            {0, 1, 2, 3, 4, 5},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
            {0, 1, 3, 4},
            {0, 1, 3, 4, 5, 6, 7, 8, 9, 10},
            {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
            {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
            {0, 1, 4, 5},
        };

        std::vector<std::string> expectedPaulis{
            "I",    "XXYZZY",     "XXYZZZZZZZZY", "XXYZZZZZZZZZZY",     "XXYZZZZZZZZZZZZZZZZY",
            "XXXX", "XXXZZZZZZX", "XXXZZZZZZZZX", "XXXZZZZZZZZZZZZZZX", "XXYY",
        };

        auto [actualCoeffs, actualPaulis, actualTargets] = hamlib::getPauliStrings(string);
        REQUIRE_THAT(actualCoeffs, Catch::Matchers::Approx(expectedCoeffs));
        REQUIRE(actualTargets == expectedTargets);
        REQUIRE(actualPaulis == expectedPaulis);
    }

    SECTION("Bravyi-Kitaev file") {

        // Excerpt of `ham_BK68` dataset from `Na2.hdf5` file from Hamlib.
        // We do not claim copyright ownership over this data. All rights remain with the respective owners.
        std::string string = "(-282.40574258226854+0j) [] +\n"
                             "(0.005017389349155809+0j) [X0 X1 X2 X3 X7 X15 Y31 Y33 X35 X39 X47] +\n"
                             "(-0.0020041447313200716+0j) [X0 X1 X2 X3 X7 X15 Y31 Y39 X47] +\n"
                             "(0.0018444840745054122+0j) [X0 X1 X2 X3 X7 X15 Y31 Z47 Z55 Z59 Y61] +\n"
                             "(0.009205462453143798+0j) [X0 X1 X2 X3 X7 Y15 Y17 X19 X23] +\n"
                             "(0.001520719168551674+0j) [X0 X1 X2 X3 X7 Y15 Z23 Y25 X27] +\n"
                             "(-0.002078812997532798+0j) [X0 X1 X2 Y3 Y5] +\n"
                             "(-9.205679278998955e-05+0j) [X0 X1 Z2 X3 X7 X15 Y31 Y32 X33 X35 X39 X47] +";

        std::vector<double> expectedCoeffs = {
            -282.40574258226854,  0.005017389349155809, -0.0020041447313200716, 0.0018444840745054122,
            0.009205462453143798, 0.001520719168551674, -0.002078812997532798,  -9.205679278998955e-05,
        };

        std::vector<std::vector<int>> expectedTargets{
            {0},
            {0, 1, 2, 3, 7, 15, 31, 33, 35, 39, 47},
            {0, 1, 2, 3, 7, 15, 31, 39, 47},
            {0, 1, 2, 3, 7, 15, 31, 47, 55, 59, 61},
            {0, 1, 2, 3, 7, 15, 17, 19, 23},
            {0, 1, 2, 3, 7, 15, 23, 25, 27},
            {0, 1, 2, 3, 5},
            {0, 1, 2, 3, 7, 15, 31, 32, 33, 35, 39, 47},
        };

        std::vector<std::string> expectedPaulis{"I",         "XXXXXXYYXXX", "XXXXXXYYX", "XXXXXXYZZZY",
                                                "XXXXXYYXX", "XXXXXYZYX",   "XXXYY",     "XXZXXXYYXXXX"};

        auto [actualCoeffs, actualPaulis, actualTargets] = hamlib::getPauliStrings(string);
        REQUIRE_THAT(actualCoeffs, Catch::Matchers::Approx(expectedCoeffs));
        REQUIRE(actualTargets == expectedTargets);
        REQUIRE(actualPaulis == expectedPaulis);
    }

    SECTION("Parity file") {
        // Excerpt of `ham_parity24` dataset from `O3.hdf5` file from Hamlib.
        // We do not claim copyright ownership over this data. All rights remain with the respective owners.
        std::string string =
            "(-208.5493426731512 [] +\n"
            "-0.005886210817222983 [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 Z21 X22] "
            "+\n"
            "-0.005886210817222983 [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 Y20 Y22 Z23] "
            "+\n"
            "0.010929140863068627 [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 Z20] +\n"
            "0.01073127560693215 [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 Z20 Z21 Z22] "
            "+\n"
            "0.004845064789709169 [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 Z20 Z22 Z23] "
            "+\n"
            "0.007513775954541585 [X0 X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 Z21] +\n"
            "4.337718262290779e-05 [X0 X1 X2 X3 X4 X5 Y6 Y12 Z13] +";

        std::vector<double> expectedCoeffs{
            -208.5493426731512,  -0.005886210817222983, -0.005886210817222983, 0.010929140863068627,
            0.01073127560693215, 0.004845064789709169,  0.007513775954541585,  4.337718262290779e-05,
        };

        std::vector<std::vector<int>> expectedTargets{
            {0},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21},
            {0, 1, 2, 3, 4, 5, 6, 12, 13},
        };

        std::vector<std::string> expectedPaulis{
            "I",
            "XXXXXXXXXXXXXXXXXXXXXZX",
            "XXXXXXXXXXXXXXXXXXXXYYZ",
            "XXXXXXXXXXXXXXXXXXXXZ",
            "XXXXXXXXXXXXXXXXXXXXZZZ",
            "XXXXXXXXXXXXXXXXXXXXZZZ",
            "XXXXXXXXXXXXXXXXXXXXZ",
            "XXXXXXYYZ",
        };

        auto [actualCoeffs, actualPaulis, actualTargets] = hamlib::getPauliStrings(string);
        REQUIRE_THAT(actualCoeffs, Catch::Matchers::Approx(expectedCoeffs));
        REQUIRE(actualTargets == expectedTargets);
        REQUIRE(actualPaulis == expectedPaulis);
    }
}
