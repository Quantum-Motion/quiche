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
 * -----------------------------------------------------------------------------
 * Portions of this file are derived from QuEST, licensed under the MIT license:
 *
 * Copyright (c) 2025 The QuEST Authors and Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "quiche/quest-patches.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include <quest/src/core/bitwise.hpp>
#include <quest/src/core/constants.hpp>
#include <quest/src/core/paulilogic.hpp>
#include <quest/src/core/validation.hpp>

// Derived from QuEST/quest/src/api/operations.cpp
// Copyright (c) 2025 The QuEST Authors and Contributors
// Licensed under the MIT License.
// Modified to implement the inverse QFT. See PR #705 for upstream patch.
void applyInverseQuantumFourierTransform(Qureg qureg, int *targets, int numTargets) {
    validate_quregFields(qureg, __func__);
    validate_targets(qureg, targets, numTargets, __func__);

    int mid = numTargets / 2; // floors
    for (int n = 0; n < mid; n++)
        applySwap(qureg, targets[n], targets[numTargets - 1 - n]);

    for (int n = 0; n < numTargets; n++) {
        for (int m = 0; m < n; m++) {
            qreal arg = -const_PI / powerOf2(m + 1);
            applyTwoQubitPhaseShift(qureg, targets[n], targets[n - m - 1], arg);
        }
        applyHadamard(qureg, targets[n]);
    }
}

void applyInverseQuantumFourierTransform(Qureg qureg, std::vector<int> targets) {
    applyInverseQuantumFourierTransform(qureg, targets.data(), targets.size());
}

std::pair<int, qreal> getMostLikelyMultiQubitOutcomeAndProb(Qureg qureg, const std::vector<int> &qubits) {
    std::vector<qreal> probs = calcProbsOfAllMultiQubitOutcomes(qureg, qubits);

    int index = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    qreal prob = probs[index];

    return {index, prob};
}

qcomp getIdentityCoeff(PauliStrSum sum) {
    // note: since QuEST allows repeated terms in PauliStrSums this
    // needs to account for the (unlikely) possibility of multiple identities

    qcomp total = 0;

    for (qindex i = 0; i < sum.numTerms; i++) {
        if (paulis_isIdentity(sum.strings[i]))
            total += sum.coeffs[i];
    }

    return total;
}

PauliStrSum cloneWithoutIdentity(PauliStrSum sum) {
    // note: since QuEST allows repeated terms in PauliStrSums this
    // needs to account for the (unlikely) possibility of multiple identities

    // these copy so initial sum is unaffected
    std::vector<qcomp> filteredCoeffs(sum.coeffs, sum.coeffs + sum.numTerms);
    std::vector<PauliStr> filteredStrings(sum.strings, sum.strings + sum.numTerms);

    for (qindex i = 0; i < filteredStrings.size(); i++) {
        if (paulis_isIdentity(filteredStrings[i])) {
            filteredStrings.erase(filteredStrings.begin() + i);
            filteredCoeffs.erase(filteredCoeffs.begin() + i);
            i--;
        }
    }

    // validates
    return createPauliStrSum(filteredStrings, filteredCoeffs);
}

void applyMultiQubitStatePhaseShift(Qureg qureg, int *targets, int *states, int numTargets, qreal angle) {
    validate_quregFields(qureg, __func__);
    validate_targets(qureg, targets, numTargets, __func__);
    validate_controlStates(states, numTargets, __func__);

    // treat as a (numTargets-1)-controlled 1-target phase shift
    qcomp phase = std::exp(1_i * angle);
    DiagMatr1 matr = (states[0] == 1) ? getDiagMatr1({1.0, phase}) : getDiagMatr1({phase, 1.0});

    int numControls = numTargets - 1;
    int *ctrlQubits = numControls > 0 ? &targets[1] : nullptr;
    int *ctrlStates = numControls > 0 ? &states[1] : nullptr;

    applyMultiStateControlledDiagMatr1(qureg, ctrlQubits, ctrlStates, numControls, targets[0], matr);
}

void applyMultiQubitStatePhaseShift(Qureg qureg, std::vector<int> targets, std::vector<int> states, qreal angle) {
    applyMultiQubitStatePhaseShift(qureg, targets.data(), states.data(), targets.size(), angle);
}

void applyMultiQubitStatePhaseFlip(Qureg qureg, int *targets, int *states, int numTargets) {
    validate_quregFields(qureg, __func__);
    validate_targets(qureg, targets, numTargets, __func__);
    validate_controlStates(states, numTargets, __func__);

    // treat as a (numTargets-1)-controlled 1-target Pauli Z
    DiagMatr1 matr = (states[0] == 1) ? getDiagMatr1({1, -1}) : getDiagMatr1({-1, 1});

    int numControls = numTargets - 1;
    int *ctrlQubits = numControls > 0 ? &targets[1] : nullptr;
    int *ctrlStates = numControls > 0 ? &states[1] : nullptr;

    applyMultiStateControlledDiagMatr1(qureg, ctrlQubits, ctrlStates, numControls, targets[0], matr);
}

void applyMultiQubitStatePhaseFlip(Qureg qureg, std::vector<int> targets, std::vector<int> states) {
    applyMultiQubitStatePhaseFlip(qureg, targets.data(), states.data(), targets.size());
}

void initClassicalState(Qureg qureg, std::vector<int> state) {
    bool isBinary = std::all_of(state.begin(), state.end(), [](int x) { return x == 0 || x == 1; });
    if (!isBinary) {
        throw std::invalid_argument("Computational basis state entries must be 0 or 1.");
    }

    int index = getIntegerFromBits(state.data(), state.size());
    initClassicalState(qureg, index);
}
