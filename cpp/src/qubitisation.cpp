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

#include <numeric>

#include <quest/src/core/bitwise.hpp>
#include <quest/src/core/constants.hpp>
#include <quest/src/core/validation.hpp>

#include "quiche/quest-patches.hpp"

namespace {

void validateQubitsUnique(std::vector<int> qubits) {
    std::sort(qubits.begin(), qubits.end());
    if (std::adjacent_find(qubits.begin(), qubits.end()) != qubits.end()) {
        throw std::invalid_argument("Qubits must be unique.");
    }
}

void validateQubitisationAncillasSize(const std::vector<int> &qubits, qindex numTerms) {
    int paddedNumQubits = std::max(std::ceil(std::log2(numTerms)), 1.0);
    if (qubits.size() < paddedNumQubits) {
        throw std::invalid_argument(
            "Number of qubits must be at least ceil(log2(numTerms)) or 1 (whichever is highest).");
    }
}

void validateCoeffsNonNegative(const std::vector<qreal> &coeffs) {
    bool nonnegative = std::all_of(coeffs.begin(), coeffs.end(), [](qreal x) { return x >= 0; });
    if (!nonnegative) {
        throw std::invalid_argument("Coefficients must be nonnegative.");
    }
}

} // namespace

namespace qubitisation {

/* SELECT ORACLE */

void applySelect(Qureg qureg, PauliStrSum sum, const std::vector<int> &indexQubits) {
    applyMultiStateControlledSelect(qureg, {}, {}, sum, indexQubits);
}

void applyControlledSelect(Qureg qureg, int control, PauliStrSum sum, const std::vector<int> &indexQubits) {
    applyMultiStateControlledSelect(qureg, {control}, {1}, sum, indexQubits);
}

void applyMultiControlledSelect(Qureg qureg, const std::vector<int> &controls, PauliStrSum sum,
                                const std::vector<int> &indexQubits) {
    std::vector<int> states(controls.size(), 1);
    applyMultiStateControlledSelect(qureg, controls, states, sum, indexQubits);
}

void applyMultiStateControlledSelect(Qureg qureg, const std::vector<int> &controls, const std::vector<int> &states,
                                     PauliStrSum sum, const std::vector<int> &indexQubits) {

    validateQubitsUnique(indexQubits);
    validateQubitisationAncillasSize(indexQubits, sum.numTerms);

    // Enforce PauliStrSum is Hermitian
    validate_pauliStrSumIsHermitian(sum, __func__);

    int numQubits = indexQubits.size();

    std::vector<int> combinedControls = indexQubits;
    combinedControls.insert(combinedControls.end(), controls.begin(), controls.end());

    std::vector<int> combinedStates(numQubits);
    combinedStates.insert(combinedStates.end(), states.begin(), states.end());

    for (int i = 0; i < sum.numTerms; i++) {
        getBitsFromInteger(combinedStates.data(), i, numQubits);
        applyMultiStateControlledPauliStr(qureg, combinedControls, combinedStates, sum.strings[i]);

        // Handle negative PauliStrSum coefficients
        // Although we could handle complex coefficients too we follow the usual approach and enforce Hermitian LCU
        double phaseAngle = std::arg(sum.coeffs[i]);
        applyMultiQubitStatePhaseShift(qureg, combinedControls, combinedStates, phaseAngle);
    }
}

/* REFLECTION */

void applyReflection(Qureg qureg, const std::vector<int> &targets) {
    applyMultiStateControlledReflection(qureg, {}, {}, targets);
}

void applyControlledReflection(Qureg qureg, int control, const std::vector<int> &targets) {
    applyMultiStateControlledReflection(qureg, {control}, {1}, targets);
}

void applyMultiControlledReflection(Qureg qureg, const std::vector<int> &controls, const std::vector<int> &targets) {
    std::vector<int> states(controls.size(), 1);
    applyMultiStateControlledReflection(qureg, controls, states, targets);
}

void applyMultiStateControlledReflection(Qureg qureg, const std::vector<int> &controls, const std::vector<int> &states,
                                         const std::vector<int> &targets) {

    validateQubitsUnique(targets);

    std::vector<int> combinedControls(targets.begin(), targets.end());
    combinedControls.insert(combinedControls.end(), controls.begin(), controls.end());

    std::vector<int> combinedStates(targets.size(), 0);
    combinedStates.insert(combinedStates.end(), states.begin(), states.end());

    // Aim: controlled reflection R =  I + 2 P_c (P_0 - I)
    // P_c = |states><states| - defined by controls, states
    // P_0 = |0...0><0...0| - defined by targets
    // I.e. when the controls are satisfied reflect about |0...0> (+1 on the zero target state,
    // -1 on every other target state); otherwise act as the identity.

    // R′ = I − 2 P_c P_0
    applyMultiQubitStatePhaseFlip(qureg, combinedControls, combinedStates);

    if (controls.empty()) {
        // No controls, so P_c = I and just need to apply -1 global phase to get R = 2 P_0 - I
        applyRotateZ(qureg, targets[0], 2 * const_PI);
    } else {
        // Flip the sign of the whole control subspace, I - 2 P_c
        // Overall yielding (I - 2 P_c)(I - 2 P_c P_0) = I - 2 P_c + 2 P_c P_0 = I + 2 P_c (P_0 - I) = R.
        applyMultiQubitStatePhaseFlip(qureg, controls, states);
    }
}

/* PREP ORACLE */

void applyCoeffsPrep(Qureg qureg, const std::vector<qreal> &coeffs, const std::vector<int> &targets, bool inverse) {
    applyMultiStateControlledCoeffsPrep(qureg, {}, {}, coeffs, targets, inverse);
}

void applyControlledCoeffsPrep(Qureg qureg, int control, const std::vector<qreal> &coeffs,
                               const std::vector<int> &targets, bool inverse) {
    applyMultiStateControlledCoeffsPrep(qureg, {control}, {1}, coeffs, targets, inverse);
}

void applyMultiControlledCoeffsPrep(Qureg qureg, std::vector<int> &controls, const std::vector<qreal> &coeffs,
                                    const std::vector<int> &targets, bool inverse) {
    std::vector<int> states(controls.size(), 1);
    applyMultiStateControlledCoeffsPrep(qureg, controls, states, coeffs, targets, inverse);
}

void applyMultiStateControlledCoeffsPrep(Qureg qureg, const std::vector<int> &controls, const std::vector<int> &states,
                                         const std::vector<qreal> &coeffs, const std::vector<int> &targets,
                                         bool inverse) {

    validateCoeffsNonNegative(coeffs);
    validateQubitsUnique(targets);
    validateQubitisationAncillasSize(targets, coeffs.size());

    int numQubits = targets.size();
    int paddedNumCoeffs = 1 << numQubits;

    std::vector<qreal> paddedCoeffs(paddedNumCoeffs, 0.0);
    std::copy(coeffs.begin(), coeffs.end(), paddedCoeffs.begin());

    qreal sum = std::accumulate(paddedCoeffs.begin(), paddedCoeffs.end(), 0.0);
    if (sum > 0.0) {
        std::transform(paddedCoeffs.begin(), paddedCoeffs.end(), paddedCoeffs.begin(),
                       [sum](qreal x) { return x / sum; });
    }

    for (int i = 0; i < numQubits; i++) {
        int j = inverse ? numQubits - i - 1 : i;

        std::vector<int> combinedControls(targets.end() - j, targets.end());
        combinedControls.insert(combinedControls.end(), controls.begin(), controls.end());

        std::vector<int> combinedStates(j);
        combinedStates.insert(combinedStates.end(), states.begin(), states.end());

        int target = targets[numQubits - j - 1];

        int numControls = 1 << j;
        int blockSize = 1 << (numQubits - j);

        for (int k = 0; k < numControls; k++) {
            int start = k * blockSize;
            int mid = start + (blockSize / 2);
            int end = start + blockSize;

            double leftSum = std::accumulate(paddedCoeffs.begin() + start, paddedCoeffs.begin() + mid, 0.0);
            double rightSum = std::accumulate(paddedCoeffs.begin() + mid, paddedCoeffs.begin() + end, 0.0);
            double denom = leftSum + rightSum;

            if (denom > 0.0) {
                double angle = (inverse ? -1 : 1) * 2.0 * std::acos(std::sqrt(leftSum / denom));
                getBitsFromInteger(combinedStates.data(), k, j);
                applyMultiStateControlledRotateY(qureg, combinedControls, combinedStates, target, angle);
            }
        }
    }
}

void applyPauliStrSumPrep(Qureg qureg, PauliStrSum sum, const std::vector<int> &targets, bool inverse) {
    applyMultiStateControlledPauliStrSumPrep(qureg, {}, {}, sum, targets, inverse);
}

void applyControlledPauliStrSumPrep(Qureg qureg, int control, PauliStrSum sum, const std::vector<int> &targets,
                                    bool inverse) {
    applyMultiStateControlledPauliStrSumPrep(qureg, {control}, {1}, sum, targets, inverse);
}

void applyMultiControlledPauliStrSumPrep(Qureg qureg, std::vector<int> &controls, PauliStrSum sum,
                                         const std::vector<int> &targets, bool inverse) {
    std::vector<int> states(controls.size(), 1);
    applyMultiStateControlledPauliStrSumPrep(qureg, controls, states, sum, targets, inverse);
}

void applyMultiStateControlledPauliStrSumPrep(Qureg qureg, const std::vector<int> &controls,
                                              const std::vector<int> &states, PauliStrSum sum,
                                              const std::vector<int> &targets, bool inverse) {
    validateQubitsUnique(targets);
    validateQubitisationAncillasSize(targets, sum.numTerms);

    // Enforce PauliStrSum is Hermitian (although abs would handle this besides negative coefficients)
    validate_pauliStrSumIsHermitian(sum, __func__);

    std::vector<qreal> magnitudes(sum.numTerms);
    std::transform(sum.coeffs, sum.coeffs + sum.numTerms, magnitudes.begin(), [](qcomp i) { return std::abs(i); });

    applyMultiStateControlledCoeffsPrep(qureg, controls, states, magnitudes, targets, inverse);
}

/* BLOCK ENCODINGS */

void applyPauliStrSumBlockEncoding(Qureg qureg, PauliStrSum sum, const std::vector<int> &indexQubits) {
    applyMultiStateControlledPauliStrSumBlockEncoding(qureg, {}, {}, sum, indexQubits);
}

void applyControlledPauliStrSumBlockEncoding(Qureg qureg, int control, PauliStrSum sum,
                                             const std::vector<int> &indexQubits) {
    applyMultiStateControlledPauliStrSumBlockEncoding(qureg, {control}, {1}, sum, indexQubits);
}

void applyMultiControlledPauliStrSumBlockEncoding(Qureg qureg, const std::vector<int> &controls, PauliStrSum sum,
                                                  const std::vector<int> &indexQubits) {
    std::vector<int> states(controls.size(), 1);
    applyMultiStateControlledPauliStrSumBlockEncoding(qureg, controls, states, sum, indexQubits);
}

void applyMultiStateControlledPauliStrSumBlockEncoding(Qureg qureg, const std::vector<int> &controls,
                                                       const std::vector<int> &states, PauliStrSum sum,
                                                       const std::vector<int> &indexQubits) {

    // validation deferred to individual methods for basic wrapper methods like this

    // note that since we choose to enforce Hermitian PauliStrSum this will also be self-inverse

    // note also that although the PREPs don't need to be controlled, as they would be computed and
    // uncomputed we simply avoid those unnecessary computations by controlling them too

    applyMultiStateControlledPauliStrSumPrep(qureg, controls, states, sum, indexQubits, false);
    applyMultiStateControlledSelect(qureg, controls, states, sum, indexQubits);
    applyMultiStateControlledPauliStrSumPrep(qureg, controls, states, sum, indexQubits, true);
}

} // namespace qubitisation
