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

#include "quiche/qdrift.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::vector<qreal> getPauliStrSumMagnitudes(PauliStrSum sum) {
    std::vector<qreal> mags(sum.numTerms);
    std::transform(sum.coeffs, sum.coeffs + sum.numTerms, mags.begin(), [](qcomp x) { return std::abs(x); });
    return mags;
}

} // namespace

namespace qdrift {

// Utils
qreal getPauliStrSumNorm(PauliStrSum sum) {
    return std::accumulate(sum.coeffs, sum.coeffs + sum.numTerms, 0.0,
                           [](qreal acc, qcomp x) { return acc + std::abs(x); });
}

// Gadgets
void applyQDRIFTMultiStateControlledPauliStrSumGadget(Qureg qureg, const std::vector<int> &controls,
                                                      const std::vector<int> &states, PauliStrSum sum, qreal angle,
                                                      int reps, std::mt19937_64 &rng) {

    // qureg, controls, states validated by applyMultiStateControlledPauliGadget
    if (reps <= 0) {
        throw std::invalid_argument("Number of reps must be positive.");
    }

    std::vector<qreal> mags = getPauliStrSumMagnitudes(sum);
    qreal norm = std::accumulate(mags.begin(), mags.end(), 0.0);
    std::discrete_distribution<int> dist(mags.begin(), mags.end());

    // -2 factor due to convention in applyPauliGadget
    qreal arg = -2 * norm * angle / reps;

    for (int i = 0; i < reps; i++) {
        int index = dist(rng);
        qreal sign = (std::real(sum.coeffs[index]) < 0) ? -1.0 : 1.0;
        applyMultiStateControlledPauliGadget(qureg, controls, states, sum.strings[index], sign * arg);
    }
}

void applyQDRIFTMultiControlledPauliStrSumGadget(Qureg qureg, const std::vector<int> &controls, PauliStrSum sum,
                                                 qreal angle, int reps, std::mt19937_64 &rng) {
    std::vector<int> states = std::vector<int>(controls.size(), 1);
    applyQDRIFTMultiStateControlledPauliStrSumGadget(qureg, controls, states, sum, angle, reps, rng);
}

void applyQDRIFTControlledPauliStrSumGadget(Qureg qureg, int control, PauliStrSum sum, qreal angle, int reps,
                                            std::mt19937_64 &rng) {
    applyQDRIFTMultiStateControlledPauliStrSumGadget(qureg, {control}, {1}, sum, angle, reps, rng);
}

void applyQDRIFTPauliStrSumGadget(Qureg qureg, PauliStrSum sum, qreal angle, int reps, std::mt19937_64 &rng) {
    applyQDRIFTMultiStateControlledPauliStrSumGadget(qureg, {}, {}, sum, angle, reps, rng);
}

// Time evolution
void applyQDRIFTUnitaryTimeEvolution(Qureg qureg, PauliStrSum hamil, qreal time, int reps, std::mt19937_64 &rng) {
    qreal angle = -time;
    applyQDRIFTPauliStrSumGadget(qureg, hamil, angle, reps, rng);
}

} // namespace qdrift
