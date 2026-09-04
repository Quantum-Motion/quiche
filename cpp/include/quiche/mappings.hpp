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
 * @file mappings.hpp
 * @brief Fermion-to-qubit mappings for simulations.
 * @author Vasco Ferreira
 */

#pragma once

#include <vector>

#include <quest.h>

namespace mappings {
std::vector<int> getQubitBasisStateJW(std::vector<int> occupation);

std::vector<int> getQubitBasisStateBK(const std::vector<int> &occupation);

std::vector<int> getQubitBasisStateParity(std::vector<int> occupation);

qindex getHartreeFockStateJW(int numElectrons);

qindex getHartreeFockStateBK(int numElectrons, int numQubits);

qindex getHartreeFockStateParity(int numElectrons, int numQubits);

} // namespace mappings
