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

#include <optional>
#include <random>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <quest.h>

#include "quiche/mappings.hpp"
#include "quiche/qpe.hpp"
#include "quiche/quest-patches.hpp"

namespace nb = nanobind;

void init_quiche_bindings(nb::module_ &m) {

    nb::module_ quiche = m.def_submodule("quiche_bindings");

    quiche.def("getHartreeFockStateJW", &mappings::getHartreeFockStateJW, nb::arg("num_electrons"));
    quiche.def("getHartreeFockStateBK", &mappings::getHartreeFockStateBK, nb::arg("num_electrons"),
               nb::arg("num_qubits"));
    quiche.def("getHartreeFockStateParity", &mappings::getHartreeFockStateParity, nb::arg("num_electrons"),
               nb::arg("num_qubits"));

    quiche.def("getPhaseTextbookTrotter", &qpe::getPhaseTextbookTrotter, nb::arg("qureg"), nb::arg("hamiltonian"),
               nb::arg("ancillas"), nb::arg("order"), nb::arg("reps"), nb::arg("time"));

    quiche.def("getPhaseKitaevTrotter", &qpe::getPhaseKitaevTrotter, nb::arg("qureg"), nb::arg("hamiltonian"),
               nb::arg("ancilla_index"), nb::arg("order"), nb::arg("reps"), nb::arg("time"), nb::arg("num_bits"));

    quiche.def(
        "getPhaseTextbookQDRIFT",
        [](Qureg qureg, PauliStrSum hamiltonian, std::vector<int> ancillas, int reps, double t,
           std::optional<unsigned int> seed) {
            std::mt19937_64 rng;

            if (seed) {
                rng = std::mt19937_64(*seed);
            } else {
                std::random_device rd;
                rng = std::mt19937_64(rd());
            }

            return qpe::getPhaseTextbookQDRIFT(qureg, hamiltonian, ancillas, reps, t, rng);
        },
        nb::arg("qureg"), nb::arg("hamiltonian"), nb::arg("ancillas"), nb::arg("reps"), nb::arg("time"),
        nb::arg("seed") = nb::none());

    quiche.def(
        "getPhaseKitaevQDRIFT",
        [](Qureg qureg, PauliStrSum hamiltonian, int ancillaIndex, int reps, double t, int numBits,
           std::optional<unsigned int> seed) {
            std::mt19937_64 rng;

            if (seed) {
                rng = std::mt19937_64(*seed);
            } else {
                std::random_device rd;
                rng = std::mt19937_64(rd());
            }

            return qpe::getPhaseKitaevQDRIFT(qureg, hamiltonian, ancillaIndex, reps, t, numBits, rng);
        },
        nb::arg("qureg"), nb::arg("hamiltonian"), nb::arg("ancilla_index"), nb::arg("reps"), nb::arg("time"),
        nb::arg("num_bits"), nb::arg("seed") = nb::none());

    quiche.def("getPhaseTextbookQubitised", &qpe::getPhaseTextbookQubitised, nb::arg("qureg"), nb::arg("hamiltonian"),
               nb::arg("qpe_ancillas"), nb::arg("qubitisation_ancillas"));

    quiche.def("getPhaseNaiveTrotter", &qpe::getPhaseNaiveTrotter, nb::arg("qureg"), nb::arg("hamiltonian"),
               nb::arg("ancilla_index"), nb::arg("order"), nb::arg("reps"), nb::arg("time"));

    quiche.def(
        "getPhaseNaiveQDRIFT",
        [](Qureg qureg, PauliStrSum hamiltonian, int ancillaIndex, int reps, double t,
           std::optional<unsigned int> seed) {
            std::mt19937_64 rng;

            if (seed) {
                rng = std::mt19937_64(*seed);
            } else {
                std::random_device rd;
                rng = std::mt19937_64(rd());
            }

            return qpe::getPhaseNaiveQDRIFT(qureg, hamiltonian, ancillaIndex, reps, t, rng);
        },
        nb::arg("qureg"), nb::arg("hamiltonian"), nb::arg("ancilla_index"), nb::arg("reps"), nb::arg("time"),
        nb::arg("seed") = nb::none());

    quiche.def("getPhaseIterativeTrotter", &qpe::getPhaseIterativeTrotter, nb::arg("qureg"), nb::arg("hamiltonian"),
               nb::arg("ancilla_index"), nb::arg("order"), nb::arg("reps"), nb::arg("time"), nb::arg("num_bits"));

    quiche.def(
        "getPhaseIterativeQDRIFT",
        [](Qureg qureg, PauliStrSum hamiltonian, int ancillaIndex, int reps, double t, int numBits,
           std::optional<unsigned int> seed) {
            std::mt19937_64 rng;

            if (seed) {
                rng = std::mt19937_64(*seed);
            } else {
                std::random_device rd;
                rng = std::mt19937_64(rd());
            }

            return qpe::getPhaseIterativeQDRIFT(qureg, hamiltonian, ancillaIndex, reps, t, numBits, rng);
        },
        nb::arg("qureg"), nb::arg("hamiltonian"), nb::arg("ancilla_index"), nb::arg("reps"), nb::arg("time"),
        nb::arg("num_bits"), nb::arg("seed") = nb::none());

    quiche.def("getPhaseTextbookQubitisedOptimised", &qpe::getPhaseTextbookQubitisedOptimised, nb::arg("qureg"),
               nb::arg("hamiltonian"), nb::arg("qpe_ancillas"), nb::arg("qubitisation_ancillas"));

    quiche.def("initClassicalState", (void (*)(Qureg, std::vector<int>))(&initClassicalState), nb::arg("qureg"),
               nb::arg("state"));
}
