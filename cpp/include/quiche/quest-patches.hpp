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
 * @file quest-patches.hpp
 * @brief Patches and utilities for QuEST.
 * @author Vasco Ferreira
 */

#ifndef QUEST_PATCHES_HPP
#define QUEST_PATCHES_HPP

#include <vector>

#include <quest.h>

// See PR 705 for Inverse QFT patch
void applyInverseQuantumFourierTransform(Qureg qureg, int *targets, int numTargets);

void applyInverseQuantumFourierTransform(Qureg qureg, std::vector<int> targets);

std::pair<int, qreal> getMostLikelyMultiQubitOutcomeAndProb(Qureg qureg, const std::vector<int> &qubits);

qcomp getIdentityCoeff(PauliStrSum sum);

PauliStrSum cloneWithoutIdentity(PauliStrSum sum);

void applyMultiStateControlledPhaseShift(Qureg qureg, int *targets, int *states, int numTargets, qreal angle);

void applyMultiStateControlledPhaseShift(Qureg qureg, std::vector<int> targets, std::vector<int> states, qreal angle);

void applyMultiStateControlledQubitPhaseFlip(Qureg qureg, int *targets, int *states, int numTargets);

void applyMultiStateControlledQubitPhaseFlip(Qureg qureg, std::vector<int> targets, std::vector<int> states);

void initClassicalState(Qureg qureg, std::vector<int> state);

#endif // QUEST_PATCHES_HPP
