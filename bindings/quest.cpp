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

#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <quest.h>
#include <quest/src/core/validation.hpp> // validateconfig_setErrorHandler

#include "resources.hpp"
#include "wrapper.hpp"

using std::string;
using std::vector;

using bindings::resources::checked;
using bindings::resources::CheckedResource;

namespace nb = nanobind;

namespace {

class QuESTError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

void QuESTErrorHandler(const char *errFunc, const char *errMsg) {
    throw QuESTError(string(errFunc) + " - " + string(errMsg));
}

// A failed QuEST initialisation leaves partial state behind, and retrying hits an
// internal error that exits the process, so retries after a failure are rejected.
bool envInitIncomplete = false;

template <class Init>
void attemptEnvInit(Init init) {
    if (envInitIncomplete)
        QuESTErrorHandler("QuESTEnv", "The QuEST environment cannot be initialised again in this process, "
                                      "because a previous initialisation failed");

    envInitIncomplete = true;
    init();
    envInitIncomplete = false;
}

} // namespace

void init_quest_bindings(nb::module_ &m) {

    nb::module_ quest = m.def_submodule("quest_bindings");

    nb::exception<QuESTError>(quest, "QuESTError", PyExc_RuntimeError);

    // QuEST's public setInputErrorHandler() unnecessarily demands that the QuESTEnv is
    // initialised before the error handler can be set. Instead use the internal setter
    // directly (skipping the validate_envIsInit check) so errors outside an active
    // QuESTEnv still throw instead of exiting the process.
    validateconfig_setErrorHandler(QuESTErrorHandler);

    // // Debug
    // nb::module_ debug = m.def_submodule("debug");
    // debug.def("setSeeds", checked(nb::overload_cast<vector<unsigned>>(&setSeeds)));
    // debug.def("getSeeds", checked(nb::overload_cast<>(&getSeeds)));
    // debug.def("setSeedsToDefault", checked(&setSeedsToDefault));
    // debug.def("getNumSeeds", checked(&getNumSeeds));
    // debug.def("setValidationOn", checked(&setValidationOn));
    // debug.def("setValidationOff", checked(&setValidationOff));
    // debug.def("setValidationEpsilonToDefault", checked(&setValidationEpsilonToDefault));
    // debug.def("setValidationEpsilon", checked(&setValidationEpsilon));
    // debug.def("getValidationEpsilon", checked(&getValidationEpsilon));
    // debug.def("setMaxNumReportedItems", checked(&setMaxNumReportedItems));
    // debug.def("setMaxNumReportedSigFigs", checked(&setMaxNumReportedSigFigs));
    // debug.def("setNumReportedNewlines", checked(&setNumReportedNewlines));
    // debug.def("setReportedPauliChars", checked(&setReportedPauliChars));
    // debug.def("setReportedPauliStrStyle", checked(&setReportedPauliStrStyle));
    // debug.def("getGpuCacheSize", checked(&getGpuCacheSize));
    // debug.def("clearGpuCache", checked(&clearGpuCache));
    // debug.def("getEnvironmentString", checked(&getEnvironmentString));

    // Env
    nb::class_<QuESTEnv>(quest, "QuESTEnv")
        .def(nb::new_([]() {
            attemptEnvInit([]() { initQuESTEnv(); });
            return getQuESTEnv();
        }))
        .def(nb::new_([](int distrib, int gpu, int multithread) {
            attemptEnvInit([=]() { initCustomQuESTEnv(distrib, gpu, multithread); });
            return getQuESTEnv();
        }))
        .def_static("report", checked(&reportQuESTEnv))
        .def("__enter__", [](QuESTEnv &self) { return self; })
        .def("__exit__",
             [](QuESTEnv, nb::args) {
                 finalizeQuESTEnv();
                 return false;
             })
        .def("sync", [](QuESTEnv &) { syncQuESTEnv(); })
        .def("isInit", [](QuESTEnv &) { return isQuESTEnvInit(); });

    // PauliStr
    nb::class_<PauliStr>(quest, "PauliStr")
        .def(nb::new_(checked(nb::overload_cast<string, vector<int>>(&getPauliStr))))
        .def(nb::new_(checked(nb::overload_cast<string>(&getPauliStr))))
        .def("report", checked(&reportPauliStr));

    // PauliStrSum
    nb::class_<CheckedResource<PauliStrSum>>(quest, "PauliStrSum")
        .def(nb::new_(checked(nb::overload_cast<vector<PauliStr>, vector<qcomp>>(&createPauliStrSum))))
        .def(nb::new_(checked(nb::overload_cast<string>(&createInlinePauliStrSum))))
        .def("destroy", &CheckedResource<PauliStrSum>::destroy)
        .def_static("fromFile", checked(nb::overload_cast<string>(&createPauliStrSumFromFile)))
        .def_static("fromReversedFile", checked(nb::overload_cast<string>(&createPauliStrSumFromReversedFile)))
        .def("report", checked(&reportPauliStrSum));

    // Qureg
    nb::class_<CheckedResource<Qureg>>(quest, "Qureg")
        .def(nb::new_(checked(&createQureg)))
        .def("destroy", &CheckedResource<Qureg>::destroy)

        // Constructors
        .def_static("createDensityQureg", checked(&createDensityQureg))
        .def_static("createForcedQureg", checked(&createForcedQureg))
        .def_static("createForcedDensityQureg", checked(&createForcedDensityQureg))
        .def_static("createCustomQureg", checked(&createCustomQureg))
        .def("createCloneQureg", checked(&createCloneQureg))

        // Initialisations
        .def("initBlankState", checked(&initBlankState))
        .def("initZeroState", checked(&initZeroState))
        .def("initPlusState", checked(&initPlusState))
        .def("initPureState", checked(&initPureState))
        .def("initClassicalState", checked(&initClassicalState))
        .def("initDebugState", checked(&initDebugState))
        .def("initArbitraryPureState", checked(&initArbitraryPureState))
        .def("initRandomPureState", checked(&initRandomPureState))
        .def("initRandomMixedState", checked(&initRandomMixedState))

        // Setters
        .def("setQuregAmps", checked(nb::overload_cast<Qureg, qindex, vector<qcomp>>(&setQuregAmps)))
        .def("setDensityQuregAmps",
             checked(nb::overload_cast<Qureg, qindex, qindex, vector<vector<qcomp>>>(&setDensityQuregAmps)))
        .def("setDensityQuregFlatAmps",
             checked(nb::overload_cast<Qureg, qindex, vector<qcomp>>(&setDensityQuregFlatAmps)))
        .def("setQuregToClone", checked(&setQuregToClone))
        .def("setQuregToWeightedSum",
             checked(nb::overload_cast<Qureg, vector<qcomp>, vector<Qureg>>(&setQuregToWeightedSum)))
        .def("setQuregToMixture", checked(nb::overload_cast<Qureg, vector<qreal>, vector<Qureg>>(&setQuregToMixture)))
        .def("setQuregToRenormalized", checked(&setQuregToRenormalized))
        .def("setQuregToPauliStrSum", checked(&setQuregToPauliStrSum))
        .def("setQuregToPartialTrace", checked(nb::overload_cast<Qureg, Qureg, vector<int>>(&setQuregToPartialTrace)))
        .def("setQuregToReducedDensityMatrix",
             checked(nb::overload_cast<Qureg, Qureg, vector<int>>(&setQuregToReducedDensityMatrix)))

        // Getters
        .def("getQuregAmps", checked(nb::overload_cast<Qureg, qindex, qindex>(&getQuregAmps)))
        .def("getDensityQuregAmps",
             checked(nb::overload_cast<Qureg, qindex, qindex, qindex, qindex>(&getDensityQuregAmps)))
        .def("getQuregAmp", checked(&getQuregAmp))
        .def("getDensityQuregAmp", checked(&getDensityQuregAmp))

        // Calculations
        .def("calcExpecPauliStr", checked(&calcExpecPauliStr))
        .def("calcExpecPauliStrSum", checked(&calcExpecPauliStrSum))
        .def("calcExpecFullStateDiagMatr", checked(&calcExpecFullStateDiagMatr))
        .def("calcExpecFullStateDiagMatrPower", checked(&calcExpecFullStateDiagMatrPower))
        .def("calcProbOfBasisState", checked(&calcProbOfBasisState))
        .def("calcProbOfQubitOutcome", checked(&calcProbOfQubitOutcome))
        .def("calcProbOfMultiQubitOutcome",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>>(&calcProbOfMultiQubitOutcome)))
        .def("calcProbsOfAllMultiQubitOutcomes",
             checked(nb::overload_cast<Qureg, vector<int>>(&calcProbsOfAllMultiQubitOutcomes)))
        .def("calcTotalProb", checked(&calcTotalProb))
        .def("calcPurity", checked(&calcPurity))
        .def("calcFidelity", checked(&calcFidelity))
        .def("calcDistance", checked(&calcDistance))
        .def("calcPartialTrace", checked(nb::overload_cast<Qureg, vector<int>>(&calcPartialTrace)))
        .def("calcReducedDensityMatrix", checked(nb::overload_cast<Qureg, vector<int>>(&calcReducedDensityMatrix)))
        .def("calcInnerProduct", checked(&calcInnerProduct))
        .def("calcExpecNonHermitianPauliStrSum", checked(&calcExpecNonHermitianPauliStrSum))
        .def("calcExpecNonHermitianFullStateDiagMatr", checked(&calcExpecNonHermitianFullStateDiagMatr))
        .def("calcExpecNonHermitianFullStateDiagMatrPower", checked(&calcExpecNonHermitianFullStateDiagMatrPower))

        // Multiplication
        .def("leftapplyCompMatr1", checked(&leftapplyCompMatr1))
        .def("rightapplyCompMatr1", checked(&rightapplyCompMatr1))
        .def("leftapplyCompMatr2", checked(&leftapplyCompMatr2))
        .def("rightapplyCompMatr2", checked(&rightapplyCompMatr2))
        .def("leftapplyCompMatr", checked(nb::overload_cast<Qureg, vector<int>, CompMatr>(&leftapplyCompMatr)))
        .def("rightapplyCompMatr", checked(nb::overload_cast<Qureg, vector<int>, CompMatr>(&rightapplyCompMatr)))
        .def("leftapplyDiagMatr1", checked(&leftapplyDiagMatr1))
        .def("rightapplyDiagMatr1", checked(&rightapplyDiagMatr1))
        .def("leftapplyDiagMatr2", checked(&leftapplyDiagMatr2))
        .def("rightapplyDiagMatr2", checked(&rightapplyDiagMatr2))
        .def("leftapplyDiagMatr", checked(nb::overload_cast<Qureg, vector<int>, DiagMatr>(&leftapplyDiagMatr)))
        .def("rightapplyDiagMatr", checked(nb::overload_cast<Qureg, vector<int>, DiagMatr>(&rightapplyDiagMatr)))
        .def("leftapplyDiagMatrPower",
             checked(nb::overload_cast<Qureg, vector<int>, DiagMatr, qcomp>(&leftapplyDiagMatrPower)))
        .def("rightapplyDiagMatrPower",
             checked(nb::overload_cast<Qureg, vector<int>, DiagMatr, qcomp>(&rightapplyDiagMatrPower)))
        .def("leftapplyFullStateDiagMatr", checked(&leftapplyFullStateDiagMatr))
        .def("rightapplyFullStateDiagMatr", checked(&rightapplyFullStateDiagMatr))
        .def("leftapplyFullStateDiagMatrPower", checked(&leftapplyFullStateDiagMatrPower))
        .def("rightapplyFullStateDiagMatrPower", checked(&rightapplyFullStateDiagMatrPower))
        .def("leftapplySwap", checked(&leftapplySwap))
        .def("rightapplySwap", checked(&rightapplySwap))
        .def("leftapplyPauliX", checked(&leftapplyPauliX))
        .def("rightapplyPauliX", checked(&rightapplyPauliX))
        .def("leftapplyPauliY", checked(&leftapplyPauliY))
        .def("rightapplyPauliY", checked(&rightapplyPauliY))
        .def("leftapplyPauliZ", checked(&leftapplyPauliZ))
        .def("rightapplyPauliZ", checked(&rightapplyPauliZ))
        .def("leftapplyPauliStr", checked(&leftapplyPauliStr))
        .def("rightapplyPauliStr", checked(&rightapplyPauliStr))
        .def("leftapplyPauliStrSum", checked(&leftapplyPauliStrSum))
        .def("rightapplyPauliStrSum", checked(&rightapplyPauliStrSum))
        .def("leftapplyPauliGadget", checked(&leftapplyPauliGadget))
        .def("rightapplyPauliGadget", checked(&rightapplyPauliGadget))
        .def("leftapplyPhaseGadget", checked(nb::overload_cast<Qureg, vector<int>, qreal>(&leftapplyPhaseGadget)))
        .def("rightapplyPhaseGadget", checked(nb::overload_cast<Qureg, vector<int>, qreal>(&rightapplyPhaseGadget)))
        .def("leftapplyMultiQubitNot", checked(nb::overload_cast<Qureg, vector<int>>(&leftapplyMultiQubitNot)))
        .def("rightapplyMultiQubitNot", checked(nb::overload_cast<Qureg, vector<int>>(&rightapplyMultiQubitNot)))
        .def("leftapplyQubitProjector", checked(&leftapplyQubitProjector))
        .def("rightapplyQubitProjector", checked(&rightapplyQubitProjector))
        // .def("leftapplyMultiQubitProjector", checked(&leftapplyMultiQubitProjector))
        // .def("rightapplyMultiQubitProjector", checked(&rightapplyMultiQubitProjector))

        // Operations
        .def("applyCompMatr1", checked(&applyCompMatr1))
        .def("applyControlledCompMatr1", checked(&applyControlledCompMatr1))
        .def("applyMultiControlledCompMatr1",
             checked(nb::overload_cast<Qureg, vector<int>, int, CompMatr1>(&applyMultiControlledCompMatr1)))
        .def("applyMultiStateControlledCompMatr1",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, CompMatr1>(
                 &applyMultiStateControlledCompMatr1)))

        .def("applyCompMatr2", checked(&applyCompMatr2))
        .def("applyControlledCompMatr2", checked(&applyControlledCompMatr2))
        .def("applyMultiControlledCompMatr2",
             checked(nb::overload_cast<Qureg, vector<int>, int, int, CompMatr2>(&applyMultiControlledCompMatr2)))
        .def("applyMultiStateControlledCompMatr2",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, int, int, CompMatr2>(
                 &applyMultiStateControlledCompMatr2)))

        .def("applyCompMatr", checked(nb::overload_cast<Qureg, vector<int>, CompMatr>(&applyCompMatr)))
        .def("applyControlledCompMatr",
             checked(nb::overload_cast<Qureg, int, vector<int>, CompMatr>(&applyControlledCompMatr)))
        .def("applyMultiControlledCompMatr",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, CompMatr>(&applyMultiControlledCompMatr)))
        .def("applyMultiStateControlledCompMatr",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, vector<int>, CompMatr>(
                 &applyMultiStateControlledCompMatr)))

        .def("applyDiagMatr1", checked(&applyDiagMatr1))
        .def("applyControlledDiagMatr1", checked(&applyControlledDiagMatr1))
        .def("applyMultiControlledDiagMatr1",
             checked(nb::overload_cast<Qureg, vector<int>, int, DiagMatr1>(&applyMultiControlledDiagMatr1)))
        .def("applyMultiStateControlledDiagMatr1",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, DiagMatr1>(
                 &applyMultiStateControlledDiagMatr1)))

        .def("applyDiagMatr2", checked(&applyDiagMatr2))
        .def("applyControlledDiagMatr2", checked(&applyControlledDiagMatr2))

        .def("applyMultiControlledDiagMatr2",
             checked(nb::overload_cast<Qureg, vector<int>, int, int, DiagMatr2>(&applyMultiControlledDiagMatr2)))
        .def("applyMultiStateControlledDiagMatr2",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, int, DiagMatr2>(
                 &applyMultiStateControlledDiagMatr2)))

        .def("applyDiagMatr", checked(nb::overload_cast<Qureg, vector<int>, DiagMatr>(&applyDiagMatr)))
        .def("applyControlledDiagMatr",
             checked(nb::overload_cast<Qureg, int, vector<int>, DiagMatr>(&applyControlledDiagMatr)))
        .def("applyMultiControlledDiagMatr",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, DiagMatr>(&applyMultiControlledDiagMatr)))
        .def("applyMultiStateControlledDiagMatr",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, vector<int>, DiagMatr>(
                 &applyMultiStateControlledDiagMatr)))
        .def("applyDiagMatrPower", checked(nb::overload_cast<Qureg, vector<int>, DiagMatr, qcomp>(&applyDiagMatrPower)))
        .def("applyControlledDiagMatrPower",
             checked(nb::overload_cast<Qureg, int, vector<int>, DiagMatr, qcomp>(&applyControlledDiagMatrPower)))
        .def("applyMultiControlledDiagMatrPower",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, DiagMatr, qcomp>(
                 &applyMultiControlledDiagMatrPower)))
        .def("applyMultiStateControlledDiagMatrPower",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, vector<int>, DiagMatr, qcomp>(
                 &applyMultiStateControlledDiagMatrPower)))

        .def("applyFullStateDiagMatr", checked(&applyFullStateDiagMatr))
        .def("applyFullStateDiagMatrPower", checked(&applyFullStateDiagMatrPower))

        .def("applyS", checked(&applyS))
        .def("applyControlledS", checked(&applyControlledS))
        .def("applyMultiControlledS", checked(nb::overload_cast<Qureg, vector<int>, int>(&applyMultiControlledS)))

        .def("applyT", checked(&applyT))
        .def("applyControlledT", checked(&applyControlledT))
        .def("applyMultiControlledT", checked(nb::overload_cast<Qureg, vector<int>, int>(&applyMultiControlledT)))
        .def("applyMultiStateControlledT",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int>(&applyMultiStateControlledT)))

        .def("applyHadamard", checked(&applyHadamard))
        .def("applyControlledHadamard", checked(&applyControlledHadamard))
        .def("applyMultiControlledHadamard",
             checked(nb::overload_cast<Qureg, vector<int>, int>(&applyMultiControlledHadamard)))
        .def("applyMultiStateControlledHadamard",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int>(&applyMultiStateControlledHadamard)))

        .def("applySwap", checked(&applySwap))
        .def("applyControlledSwap", checked(&applyControlledSwap))
        .def("applyMultiControlledSwap",
             checked(nb::overload_cast<Qureg, vector<int>, int, int>(&applyMultiControlledSwap)))
        .def("applyMultiStateControlledSwap",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, int>(&applyMultiStateControlledSwap)))

        .def("applySqrtSwap", checked(&applySqrtSwap))
        .def("applyControlledSqrtSwap", checked(&applyControlledSqrtSwap))
        .def("applyMultiControlledSqrtSwap",
             checked(nb::overload_cast<Qureg, vector<int>, int, int>(&applyMultiControlledSqrtSwap)))
        // .def("applyMultiStateControlledSqrtSwap", checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int,
        // int, int>(&applyMultiStateControlledSqrtSwap)))

        .def("applyPauliX", checked(&applyPauliX))
        .def("applyPauliY", checked(&applyPauliY))
        .def("applyPauliZ", checked(&applyPauliZ))
        .def("applyControlledPauliX", checked(&applyControlledPauliX))
        .def("applyControlledPauliY", checked(&applyControlledPauliY))
        .def("applyControlledPauliZ", checked(&applyControlledPauliZ))

        .def("applyMultiControlledPauliX",
             checked(nb::overload_cast<Qureg, vector<int>, int>(&applyMultiControlledPauliX)))
        .def("applyMultiControlledPauliY",
             checked(nb::overload_cast<Qureg, vector<int>, int>(&applyMultiControlledPauliY)))
        .def("applyMultiControlledPauliZ",
             checked(nb::overload_cast<Qureg, vector<int>, int>(&applyMultiControlledPauliZ)))
        .def("applyMultiStateControlledPauliX",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int>(&applyMultiStateControlledPauliX)))
        .def("applyMultiStateControlledPauliY",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int>(&applyMultiStateControlledPauliY)))
        .def("applyMultiStateControlledPauliZ",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int>(&applyMultiStateControlledPauliZ)))

        .def("applyPauliStr", checked(&applyPauliStr))
        .def("applyControlledPauliStr", checked(&applyControlledPauliStr))
        .def("applyMultiControlledPauliStr",
             checked(nb::overload_cast<Qureg, vector<int>, PauliStr>(&applyMultiControlledPauliStr)))
        .def("applyMultiStateControlledPauliStr",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, PauliStr>(&applyMultiStateControlledPauliStr)))

        .def("applyRotateX", checked(&applyRotateX))
        .def("applyRotateY", checked(&applyRotateY))
        .def("applyRotateZ", checked(&applyRotateZ))
        .def("applyControlledRotateX", checked(&applyControlledRotateX))
        .def("applyControlledRotateY", checked(&applyControlledRotateY))
        .def("applyControlledRotateZ", checked(&applyControlledRotateZ))

        .def("applyMultiControlledRotateX",
             checked(nb::overload_cast<Qureg, vector<int>, int, qreal>(&applyMultiControlledRotateX)))
        .def("applyMultiControlledRotateY",
             checked(nb::overload_cast<Qureg, vector<int>, int, qreal>(&applyMultiControlledRotateY)))
        .def("applyMultiControlledRotateZ",
             checked(nb::overload_cast<Qureg, vector<int>, int, qreal>(&applyMultiControlledRotateZ)))
        .def("applyMultiStateControlledRotateX",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, qreal>(&applyMultiStateControlledRotateX)))
        .def("applyMultiStateControlledRotateY",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, qreal>(&applyMultiStateControlledRotateY)))
        .def("applyMultiStateControlledRotateZ",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, qreal>(&applyMultiStateControlledRotateZ)))

        .def("applyRotateAroundAxis", checked(&applyRotateAroundAxis))
        .def("applyControlledRotateAroundAxis", checked(&applyControlledRotateAroundAxis))
        .def("applyMultiControlledRotateAroundAxis",
             checked(nb::overload_cast<Qureg, vector<int>, int, qreal, qreal, qreal, qreal>(
                 &applyMultiControlledRotateAroundAxis)))
        .def("applyMultiStateControlledRotateAroundAxis",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, int, qreal, qreal, qreal, qreal>(
                 &applyMultiStateControlledRotateAroundAxis)))

        .def("applyPauliGadget", checked(&applyPauliGadget))
        .def("applyNonUnitaryPauliGadget", checked(&applyNonUnitaryPauliGadget))
        .def("applyControlledPauliGadget", checked(&applyControlledPauliGadget))

        .def("applyMultiControlledPauliGadget",
             checked(nb::overload_cast<Qureg, vector<int>, PauliStr, qreal>(&applyMultiControlledPauliGadget)))
        .def("applyMultiStateControlledPauliGadget",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, PauliStr, qreal>(
                 &applyMultiStateControlledPauliGadget)))

        .def("applyPhaseGadget", checked(nb::overload_cast<Qureg, vector<int>, qreal>(&applyPhaseGadget)))
        .def("applyControlledPhaseGadget",
             checked(nb::overload_cast<Qureg, int, vector<int>, qreal>(&applyControlledPhaseGadget)))

        .def("applyMultiControlledPhaseGadget",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, qreal>(&applyMultiControlledPhaseGadget)))
        .def("applyMultiStateControlledPhaseGadget",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, vector<int>, qreal>(
                 &applyMultiStateControlledPhaseGadget)))

        .def("applyPhaseFlip", checked(&applyPhaseFlip))
        .def("applyPhaseShift", checked(&applyPhaseShift))
        .def("applyTwoQubitPhaseFlip", checked(&applyTwoQubitPhaseFlip))
        .def("applyTwoQubitPhaseShift", checked(&applyTwoQubitPhaseShift))

        .def("applyMultiQubitPhaseFlip", checked(nb::overload_cast<Qureg, vector<int>>(&applyMultiQubitPhaseFlip)))
        .def("applyMultiQubitPhaseShift",
             checked(nb::overload_cast<Qureg, vector<int>, qreal>(&applyMultiQubitPhaseShift)))
        .def("applyMultiQubitNot", checked(nb::overload_cast<Qureg, vector<int>>(&applyMultiQubitNot)))
        .def("applyControlledMultiQubitNot",
             checked(nb::overload_cast<Qureg, int, vector<int>>(&applyControlledMultiQubitNot)))
        .def("applyMultiControlledMultiQubitNot",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>>(&applyMultiControlledMultiQubitNot)))
        .def("applyMultiStateControlledMultiQubitNot",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, vector<int>>(
                 &applyMultiStateControlledMultiQubitNot)))

        .def("applyQubitMeasurement", checked(&applyQubitMeasurement))
        .def("applyForcedQubitMeasurement", checked(&applyForcedQubitMeasurement))
        // applyMultiQubitMeasurementAndGetProb
        .def("applyForcedMultiQubitMeasurement",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>>(&applyForcedMultiQubitMeasurement)))

        .def("applyQubitProjector", checked(&applyQubitProjector))
        .def("applyMultiQubitProjector",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>>(&applyMultiQubitProjector)))

        .def("applyQuantumFourierTransform",
             checked(nb::overload_cast<Qureg, vector<int>>(&applyQuantumFourierTransform)))
        .def("applyFullQuantumFourierTransform", checked(&applyFullQuantumFourierTransform))

        // Decoherence
        .def("mixDephasing", checked(&mixDephasing))
        .def("mixTwoQubitDephasing", checked(&mixTwoQubitDephasing))
        .def("mixDepolarising", checked(&mixDepolarising))
        .def("mixTwoQubitDepolarising", checked(&mixTwoQubitDepolarising))
        .def("mixDamping", checked(&mixDamping))
        .def("mixPaulis", checked(&mixPaulis))
        .def("mixQureg", checked(&mixQureg))
        .def("mixKrausMap", checked(nb::overload_cast<Qureg, vector<int>, KrausMap>(&mixKrausMap)))
        .def("mixSuperOp", checked(nb::overload_cast<Qureg, vector<int>, SuperOp>(&mixSuperOp)))

        // Trotterisation
        .def("applyTrotterizedPauliStrSumGadget", checked(&applyTrotterizedPauliStrSumGadget))
        .def("applyTrotterizedControlledPauliStrSumGadget", checked(&applyTrotterizedControlledPauliStrSumGadget))
        .def("applyTrotterizedMultiControlledPauliStrSumGadget",
             checked(nb::overload_cast<Qureg, vector<int>, PauliStrSum, qreal, int, int>(
                 &applyTrotterizedMultiControlledPauliStrSumGadget)))
        .def("applyTrotterizedMultiStateControlledPauliStrSumGadget",
             checked(nb::overload_cast<Qureg, vector<int>, vector<int>, PauliStrSum, qreal, int, int>(
                 &applyTrotterizedMultiStateControlledPauliStrSumGadget)))
        .def("applyTrotterizedNonUnitaryPauliStrSumGadget", checked(&applyTrotterizedNonUnitaryPauliStrSumGadget))
        .def("applyTrotterizedUnitaryTimeEvolution", checked(&applyTrotterizedUnitaryTimeEvolution))
        .def("applyTrotterizedImaginaryTimeEvolution", checked(&applyTrotterizedImaginaryTimeEvolution))
        // .def("applyTrotterizedNoisyTimeEvolution", checked(&applyTrotterizedNoisyTimeEvolution))

        // Syncing
        .def("syncQuregToGpu", checked(&syncQuregToGpu))
        .def("syncQuregFromGpu", checked(&syncQuregFromGpu))
        .def("syncSubQuregToGpu", checked(&syncSubQuregToGpu))
        .def("syncSubQuregFromGpu", checked(&syncSubQuregFromGpu))

        // Reporters
        .def("reportQuregParams", checked(&reportQuregParams))
        .def("report", checked(&reportQureg));

    // KrausMap
    nb::class_<CheckedResource<KrausMap>>(quest, "KrausMap")
        .def(nb::new_(checked(&createKrausMap)))
        .def(nb::new_(checked(&createInlineKrausMap)))
        .def("destroy", &CheckedResource<KrausMap>::destroy)
        .def("set", checked(nb::overload_cast<KrausMap, vector<vector<vector<qcomp>>>>(&setKrausMap)))
        .def("sync", checked(&syncKrausMap))
        .def("report", checked(&reportKrausMap));

    // SuperOp
    nb::class_<CheckedResource<SuperOp>>(quest, "SuperOp")
        .def(nb::new_(checked(&createSuperOp)))
        .def(nb::new_(checked(&createInlineSuperOp)))
        .def("destroy", &CheckedResource<SuperOp>::destroy)
        .def("set", checked(nb::overload_cast<SuperOp, vector<vector<qcomp>>>(&setSuperOp)))
        .def("sync", checked(&syncSuperOp))
        .def("report", checked(&reportSuperOp));

    // CompMatr1
    nb::class_<CompMatr1>(quest, "CompMatr1")
        .def(nb::new_(checked(nb::overload_cast<vector<vector<qcomp>>>(&getCompMatr1))))
        .def("report", checked(&reportCompMatr1));

    // CompMatr2
    nb::class_<CompMatr2>(quest, "CompMatr2")
        .def(nb::new_(checked(nb::overload_cast<vector<vector<qcomp>>>(&getCompMatr2))))
        .def("report", checked(&reportCompMatr2));

    // CompMatr
    nb::class_<CheckedResource<CompMatr>>(quest, "CompMatr")
        .def(nb::new_(checked(&createCompMatr)))
        .def("destroy", &CheckedResource<CompMatr>::destroy)
        .def("set", checked(nb::overload_cast<CompMatr, vector<vector<qcomp>>>(&setCompMatr)))
        .def("sync", checked(&syncCompMatr))
        .def("report", checked(&reportCompMatr));

    // DiagMatr1
    nb::class_<DiagMatr1>(quest, "DiagMatr1")
        .def(nb::new_(checked(nb::overload_cast<vector<qcomp>>(&getDiagMatr1))))
        .def("report", checked(&reportDiagMatr1));

    // DiagMatr2
    nb::class_<DiagMatr2>(quest, "DiagMatr2")
        .def(nb::new_(checked(nb::overload_cast<vector<qcomp>>(&getDiagMatr2))))
        .def("report", checked(&reportDiagMatr2));

    // DiagMatr
    nb::class_<CheckedResource<DiagMatr>>(quest, "DiagMatr")
        .def(nb::new_(checked(&createDiagMatr)))
        .def("destroy", &CheckedResource<DiagMatr>::destroy)
        .def("set", checked(nb::overload_cast<DiagMatr, vector<qcomp>>(&setDiagMatr)))
        .def("sync", checked(&syncDiagMatr))
        .def("report", checked(&reportDiagMatr));

    // FullStateDiagMatr
    nb::class_<CheckedResource<FullStateDiagMatr>>(quest, "FullStateDiagMatr")
        .def(nb::new_(checked(&createFullStateDiagMatr)))
        .def("destroy", &CheckedResource<FullStateDiagMatr>::destroy)
        .def("set", checked(nb::overload_cast<FullStateDiagMatr, qindex, vector<qcomp>>(&setFullStateDiagMatr)))
        .def("sync", checked(&syncFullStateDiagMatr))
        .def("report", checked(&reportFullStateDiagMatr));
}
