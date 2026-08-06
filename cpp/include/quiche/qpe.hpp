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

/**
 * @file qpe.hpp
 * @brief Quantum phase estimation algorithms.
 * @author Vasco Ferreira
 */

#ifndef QPE_HPP
#define QPE_HPP

#include <random>
#include <vector>

#include <quest.h>

namespace qpe {

double getPhaseTextbookTrotter(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &ancillas, int order,
                               int reps, double t);

double getPhaseTextbookQDRIFT(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &ancillas, int reps,
                              double t, std::mt19937_64 &rng);

double getPhaseTextbookQubitised(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &qpeAncillas,
                                 const std::vector<int> &qubitisationAncillas);

double getPhaseKitaevTrotter(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int order, int reps, double t,
                             int numBits);

double getPhaseKitaevQDRIFT(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int reps, double t, int numBits,
                            std::mt19937_64 &rng);

double getPhaseNaiveTrotter(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int order, int reps, double t);

double getPhaseNaiveQDRIFT(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int reps, double t, std::mt19937_64 &rng);

double getPhaseIterativeTrotter(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int order, int reps, double t,
                                int numBits);

double getPhaseIterativeQDRIFT(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int reps, double t, int numBits,
                               std::mt19937_64 &rng);

double getPhaseTextbookQubitisedOptimised(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &qpeAncillas,
                                          const std::vector<int> &qubitisationAncillas);

} // namespace qpe

#endif // QPE_HPP
