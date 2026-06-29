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
 * @file qubitisation.hpp
 * @brief Block encoding primitives for qubitisation.
 * @author Vasco Ferreira
 */

#ifndef QUBITISATION_HPP
#define QUBITISATION_HPP

#include <quest.h>

namespace qubitisation {

/* SELECT ORACLE */

void applySelect(Qureg qureg, PauliStrSum sum, const std::vector<int> &indexQubits);

void applyControlledSelect(Qureg qureg, int control, PauliStrSum sum, const std::vector<int> &indexQubits);

void applyMultiControlledSelect(Qureg qureg, const std::vector<int> &controls, PauliStrSum sum,
                                const std::vector<int> &indexQubits);

void applyMultiStateControlledSelect(Qureg qureg, const std::vector<int> &controls, const std::vector<int> &states,
                                     PauliStrSum sum, const std::vector<int> &indexQubits);

/* REFLECTION */

void applyReflection(Qureg qureg, const std::vector<int> &targets);

void applyControlledReflection(Qureg qureg, int control, const std::vector<int> &targets);

void applyMultiControlledReflection(Qureg qureg, const std::vector<int> &controls, const std::vector<int> &targets);

void applyMultiStateControlledReflection(Qureg qureg, const std::vector<int> &controls, const std::vector<int> &states,
                                         const std::vector<int> &targets);

/* PREP ORACLE */
void applyCoeffsPrep(Qureg qureg, const std::vector<qreal> &coeffs, const std::vector<int> &targets, bool inverse);

void applyControlledCoeffsPrep(Qureg qureg, int control, const std::vector<qreal> &coeffs,
                               const std::vector<int> &targets, bool inverse);

void applyMultiControlledCoeffsPrep(Qureg qureg, std::vector<int> &controls, const std::vector<qreal> &coeffs,
                                    const std::vector<int> &targets, bool inverse);

void applyMultiStateControlledCoeffsPrep(Qureg qureg, const std::vector<int> &controls, const std::vector<int> &states,
                                         const std::vector<qreal> &coeffs, const std::vector<int> &targets,
                                         bool inverse);

void applyPauliStrSumPrep(Qureg qureg, PauliStrSum sum, const std::vector<int> &targets, bool inverse);

void applyControlledPauliStrSumPrep(Qureg qureg, int control, PauliStrSum sum, const std::vector<int> &targets,
                                    bool inverse);

void applyMultiControlledPauliStrSumPrep(Qureg qureg, std::vector<int> &controls, PauliStrSum sum,
                                         const std::vector<int> &targets, bool inverse);

void applyMultiStateControlledPauliStrSumPrep(Qureg qureg, const std::vector<int> &controls,
                                              const std::vector<int> &states, PauliStrSum sum,
                                              const std::vector<int> &targets, bool inverse);

/* BLOCK ENCODINGS */

void applyPauliStrSumBlockEncoding(Qureg qureg, PauliStrSum sum, const std::vector<int> &indexQubits);

void applyControlledPauliStrSumBlockEncoding(Qureg qureg, int control, PauliStrSum sum,
                                             const std::vector<int> &indexQubits);

void applyMultiControlledPauliStrSumBlockEncoding(Qureg qureg, const std::vector<int> &controls, PauliStrSum sum,
                                                  const std::vector<int> &indexQubits);

void applyMultiStateControlledPauliStrSumBlockEncoding(Qureg qureg, const std::vector<int> &controls,
                                                       const std::vector<int> &states, PauliStrSum sum,
                                                       const std::vector<int> &indexQubits);

} // namespace qubitisation

#endif // QUBITISATION_HPP
