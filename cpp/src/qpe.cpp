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

#include "quiche/qpe.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <quest/src/core/constants.hpp>

#include "quiche/qdrift.hpp"
#include "quiche/qubitisation.hpp"
#include "quiche/quest-patches.hpp"

// Note we make no assumptions about the ordering, or contiguity of ancillas. They are
// only required to be within the range of the Qureg and not overlap with the data.
// Actually we also assume that they are unique e.g. can't have the same qubit twice.

namespace {

template <typename F>
double getPhaseTextbookInner(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &ancillas,
                             F applyControlledUnitary) {

    for (int index = 0; index < ancillas.size(); index++) {
        int ancilla = ancillas[index];

        // Ancilla qubits state prep
        applyHadamard(qureg, ancilla);

        applyControlledUnitary(qureg, ancilla, hamiltonian, index);
    }

    applyInverseQuantumFourierTransform(qureg, ancillas);

    // Get most likely outcome (deterministic)
    auto [index, prob] = getMostLikelyMultiQubitOutcomeAndProb(qureg, ancillas);
    double phase = static_cast<double>(index) / (1 << ancillas.size());

    // Wrap phase from [0, 1] to [-0.5, 0.5]
    if (phase > 0.5) {
        phase -= 1.0;
    }

    return phase;
}

template <typename F>
double getPhaseKitaevInner(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int numBits, F applyControlledUnitary) {

    auto circularDistance = [](double x, double y) {
        double diff = std::abs(x - y);
        return std::min(diff, 1.0 - diff);
    };

    double phase = 0.0;
    for (int index = numBits - 1; index >= 0; index--) {

        // Ancilla qubit state prep
        applyForcedQubitMeasurement(qureg, ancilla, 0);
        applyHadamard(qureg, ancilla);

        applyControlledUnitary(qureg, ancilla, hamiltonian, index);

        double expecX = calcExpecPauliStr(qureg, getPauliStr("X", {ancilla}));
        double expecY = calcExpecPauliStr(qureg, getPauliStr("Y", {ancilla}));
        double theta = std::atan2(expecY, expecX) / (2.0 * const_PI); // [-0.5, 0.5]

        // Wrap from [0, 1]
        if (theta < 0) {
            theta += 1.0;
        }

        if (index == numBits - 1) {
            phase = theta;
        } else {
            double zeroCase = phase / 2.0;
            double oneCase = (phase + 1.0) / 2.0;
            phase = circularDistance(zeroCase, theta) < circularDistance(oneCase, theta) ? zeroCase : oneCase;
        }
    }

    // Wrap phase from [0, 1] to [-0.5, 0.5]
    if (phase > 0.5) {
        phase -= 1.0;
    }

    return phase;
}

template <typename F>
double getPhaseNaiveInner(Qureg qureg, PauliStrSum hamiltonian, int ancilla, F applyControlledUnitary) {

    applyHadamard(qureg, ancilla);

    applyControlledUnitary(qureg, ancilla, hamiltonian);

    // Determine phase
    double expecX = calcExpecPauliStr(qureg, getPauliStr("X", {ancilla}));
    double expecY = calcExpecPauliStr(qureg, getPauliStr("Y", {ancilla}));
    double phase = std::atan2(expecY, expecX) / (2 * const_PI); // [-0.5, 0.5]

    return phase;
}

template <typename F>
double getPhaseIterativeInner(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int numBits,
                              F applyControlledUnitary) {

    double phase = 0.0;
    for (int index = numBits - 1; index >= 0; index--) {

        // Ancilla qubit state prep
        applyForcedQubitMeasurement(qureg, ancilla, 0);
        applyHadamard(qureg, ancilla);

        applyControlledUnitary(qureg, ancilla, hamiltonian, index);

        // Phase feedback
        applyPhaseShift(qureg, ancilla, -2.0 * const_PI * phase * (1 << index));

        // Mimic measurement
        applyHadamard(qureg, ancilla);
        double p1 = calcProbOfQubitOutcome(qureg, ancilla, 1);
        applyHadamard(qureg, ancilla);

        if (p1 >= 0.5) {
            phase += 1.0 / (1 << (index + 1));
        }
    }

    // Wrap phase from [0, 1] to [-0.5, 0.5]
    if (phase > 0.5) {
        phase -= 1.0;
    }

    return phase;
}

} // namespace

namespace qpe {

double getPhaseTextbookTrotter(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &ancillas, int order,
                               int reps, double t) {

    auto trotterLambda = [order, reps, t](Qureg qureg, int control, PauliStrSum hamiltonian, int index) {
        int steps = 1 << index;
        applyTrotterizedControlledPauliStrSumGadget(qureg, control, hamiltonian, t * steps, order, reps * steps);
    };

    return getPhaseTextbookInner(qureg, hamiltonian, ancillas, trotterLambda);
}

double getPhaseTextbookQDRIFT(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &ancillas, int reps,
                              double t, std::mt19937_64 &rng) {

    auto trotterLambda = [reps, t, &rng](Qureg qureg, int control, PauliStrSum hamiltonian, int index) {
        int steps = 1 << index;
        qdrift::applyQDRIFTControlledPauliStrSumGadget(qureg, control, hamiltonian, t * steps, reps * steps, rng);
    };

    return getPhaseTextbookInner(qureg, hamiltonian, ancillas, trotterLambda);
}

double getPhaseKitaevTrotter(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int order, int reps, double t,
                             int numBits) {

    auto trotterLambda = [order, reps, t](Qureg qureg, int control, PauliStrSum hamiltonian, int index) {
        int steps = 1 << index;
        applyTrotterizedControlledPauliStrSumGadget(qureg, control, hamiltonian, t * steps, order, reps * steps);
    };

    return getPhaseKitaevInner(qureg, hamiltonian, ancilla, numBits, trotterLambda);
}

double getPhaseKitaevQDRIFT(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int reps, double t, int numBits,
                            std::mt19937_64 &rng) {

    auto trotterLambda = [reps, t, &rng](Qureg qureg, int control, PauliStrSum hamiltonian, int index) {
        int steps = 1 << index;
        qdrift::applyQDRIFTControlledPauliStrSumGadget(qureg, control, hamiltonian, t * steps, reps * steps, rng);
    };

    return getPhaseKitaevInner(qureg, hamiltonian, ancilla, numBits, trotterLambda);
}

double getPhaseTextbookQubitised(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &qpeAncillas,
                                 const std::vector<int> &qubitisationAncillas) {

    // Naive method
    auto qubitisationLambda = [&qubitisationAncillas](Qureg qureg, int control, PauliStrSum hamiltonian, int index) {
        int steps = 1 << index;
        for (int j = 0; j < steps; j++) {
            qubitisation::applyControlledPauliStrSumBlockEncoding(qureg, control, hamiltonian, qubitisationAncillas);
            qubitisation::applyControlledReflection(qureg, control, qubitisationAncillas);
        }
    };

    return getPhaseTextbookInner(qureg, hamiltonian, qpeAncillas, qubitisationLambda);
}

double getPhaseNaiveTrotter(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int order, int reps, double t) {

    auto trotterLambda = [order, reps, t](Qureg qureg, int control, PauliStrSum hamiltonian) {
        applyTrotterizedControlledPauliStrSumGadget(qureg, control, hamiltonian, t, order, reps);
    };

    return getPhaseNaiveInner(qureg, hamiltonian, ancilla, trotterLambda);
}

double getPhaseNaiveQDRIFT(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int reps, double t,
                           std::mt19937_64 &rng) {

    auto trotterLambda = [reps, t, &rng](Qureg qureg, int control, PauliStrSum hamiltonian) {
        qdrift::applyQDRIFTControlledPauliStrSumGadget(qureg, control, hamiltonian, t, reps, rng);
    };

    return getPhaseNaiveInner(qureg, hamiltonian, ancilla, trotterLambda);
}

double getPhaseIterativeTrotter(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int order, int reps, double t,
                                int numBits) {

    auto trotterLambda = [order, reps, t](Qureg qureg, int control, PauliStrSum hamiltonian, int index) {
        int steps = 1 << index;
        applyTrotterizedControlledPauliStrSumGadget(qureg, control, hamiltonian, t * steps, order, reps * steps);
    };

    return getPhaseIterativeInner(qureg, hamiltonian, ancilla, numBits, trotterLambda);
}

double getPhaseIterativeQDRIFT(Qureg qureg, PauliStrSum hamiltonian, int ancilla, int reps, double t, int numBits,
                               std::mt19937_64 &rng) {

    auto trotterLambda = [reps, t, &rng](Qureg qureg, int control, PauliStrSum hamiltonian, int index) {
        int steps = 1 << index;
        qdrift::applyQDRIFTControlledPauliStrSumGadget(qureg, control, hamiltonian, t * steps, reps * steps, rng);
    };

    return getPhaseIterativeInner(qureg, hamiltonian, ancilla, numBits, trotterLambda);
}

double getPhaseTextbookQubitisedOptimised(Qureg qureg, PauliStrSum hamiltonian, const std::vector<int> &qpeAncillas,
                                          const std::vector<int> &qubitisationAncillas) {
    // Optimised Qubitisation QPE method from Babbush, et. al. (arXiv:1805.03662)

    // R_L = PREP · R0 · PREP†
    auto applyMultiStateControlledReflection = [&qubitisationAncillas](Qureg qureg, const std::vector<int> &controls,
                                                                       const std::vector<int> &states,
                                                                       PauliStrSum hamiltonian) {
        qubitisation::applyPauliStrSumPrep(qureg, hamiltonian, qubitisationAncillas, true);
        qubitisation::applyMultiStateControlledReflection(qureg, controls, states, qubitisationAncillas);
        qubitisation::applyPauliStrSumPrep(qureg, hamiltonian, qubitisationAncillas, false);
    };

    // W = R_L · SELECT
    auto applyMultiStateControlledWalk = [applyMultiStateControlledReflection,
                                          &qubitisationAncillas](Qureg qureg, const std::vector<int> &controls,
                                                                 const std::vector<int> &states,
                                                                 PauliStrSum hamiltonian) {
        qubitisation::applyMultiStateControlledSelect(qureg, controls, states, hamiltonian, qubitisationAncillas);
        applyMultiStateControlledReflection(qureg, controls, states, hamiltonian);
    };

    auto qubitisationLambda = [applyMultiStateControlledReflection,
                               applyMultiStateControlledWalk](Qureg qureg, int control, PauliStrSum hamiltonian,
                                                              int index) {
        if (index == 0) {
            applyMultiStateControlledWalk(qureg, {control}, {1}, hamiltonian);
        } else {
            applyMultiStateControlledReflection(qureg, {control}, {0}, hamiltonian);

            int steps = 1 << (index - 1);
            for (int j = 0; j < steps; j++) {
                applyMultiStateControlledWalk(qureg, {}, {}, hamiltonian);
            }

            applyMultiStateControlledReflection(qureg, {control}, {0}, hamiltonian);
        }
    };

    // Conventional walk operator operates in subspace spanned by {|0>|k>, |phi ⊥>}
    // This modified walk operator operates in subspace spanned the {|L>|k>, |psi ⊥>}
    // Thus need PREP on qubitisationAncillas
    qubitisation::applyPauliStrSumPrep(qureg, hamiltonian, qubitisationAncillas, false);

    return getPhaseTextbookInner(qureg, hamiltonian, qpeAncillas, qubitisationLambda);
}

} // namespace qpe
