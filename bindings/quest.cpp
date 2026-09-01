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

using std::string;
using std::vector;

namespace nb = nanobind;

void errorHandler(const char *errFunc, const char *errMsg) {
    throw std::runtime_error(string(errFunc) + " - " + string(errMsg));
}

void init_quest_bindings(nb::module_ &m) {

    nb::module_ quest = m.def_submodule("quest_bindings");

    // // Debug
    // nb::module_ debug = m.def_submodule("debug");
    // debug.def("setSeeds", (void (*)(vector<unsigned>))(&setSeeds));
    // debug.def("getSeeds", (vector<unsigned>(*)())(&getSeeds));
    // debug.def("setSeedsToDefault", (&setSeedsToDefault));
    // debug.def("getNumSeeds", (&getNumSeeds));
    // debug.def("setValidationOn", (&setValidationOn));
    // debug.def("setValidationOff", (&setValidationOff));
    // debug.def("setValidationEpsilonToDefault", (&setValidationEpsilonToDefault));
    // debug.def("setValidationEpsilon", (&setValidationEpsilon));
    // debug.def("getValidationEpsilon", (&getValidationEpsilon));
    // debug.def("setMaxNumReportedItems", (&setMaxNumReportedItems));
    // debug.def("setMaxNumReportedSigFigs", (&setMaxNumReportedSigFigs));
    // debug.def("setNumReportedNewlines", (&setNumReportedNewlines));
    // debug.def("setReportedPauliChars", (&setReportedPauliChars));
    // debug.def("setReportedPauliStrStyle", (&setReportedPauliStrStyle));
    // debug.def("getGpuCacheSize", (&getGpuCacheSize));
    // debug.def("clearGpuCache", (&clearGpuCache));
    // // debug.def("getEnvironmentString", (&getEnvironmentString));

    // Env
    nb::class_<QuESTEnv>(quest, "QuESTEnv")
        .def(nb::new_([]() {
            initQuESTEnv();
            setInputErrorHandler(errorHandler);
            return getQuESTEnv();
        }))
        .def(nb::new_([](int distrib, int gpu, int multithread) {
            initCustomQuESTEnv(distrib, gpu, multithread);
            setInputErrorHandler(errorHandler);
            return getQuESTEnv();
        }))
        .def_static("report", &reportQuESTEnv)
        .def("__enter__", [](QuESTEnv &self) { return self; })
        .def("__exit__",
             [](QuESTEnv, nb::args) {
                 finalizeQuESTEnv();
                 return false;
             })
        .def("syncQuESTEnv", (&syncQuESTEnv))
        .def("isQuESTEnvInit", (&isQuESTEnvInit));

    // PauliStr
    nb::class_<PauliStr>(quest, "PauliStr")
        .def(nb::new_((PauliStr (*)(string, vector<int>))(&getPauliStr)))
        .def(nb::new_((PauliStr (*)(string))(&getPauliStr)))
        .def("report", &reportPauliStr);

    // PauliStrSum
    nb::class_<PauliStrSum>(quest, "PauliStrSum")
        .def(nb::new_((PauliStrSum (*)(vector<PauliStr>, vector<qcomp>))(&createPauliStrSum)))
        .def(nb::new_((PauliStrSum (*)(string))(&createInlinePauliStrSum)))
        .def("__del__", &destroyPauliStrSum)
        .def("fromFile", (PauliStrSum (*)(string))(&createPauliStrSumFromFile))
        .def("fromReversedFile", (PauliStrSum (*)(string))(&createPauliStrSumFromReversedFile))
        .def("report", reportPauliStrSum);

    // Qureg
    nb::class_<Qureg>(quest, "Qureg")
        .def(nb::new_(&createQureg))
        .def("__del__", &destroyQureg)

        // Constructors
        .def("createDensityQureg", (&createDensityQureg))
        .def("createForcedQureg", (&createForcedQureg))
        .def("createForcedDensityQureg", (&createForcedDensityQureg))
        .def("createCustomQureg", (&createCustomQureg))
        .def("createCloneQureg", (&createCloneQureg))

        // Initialisations
        .def("initBlankState", &initBlankState)
        .def("initBlankState", &initBlankState)
        .def("initZeroState", &initZeroState)
        .def("initPlusState", &initPlusState)
        .def("initPureState", &initPureState)
        .def("initClassicalState", &initClassicalState)
        .def("initDebugState", &initDebugState)
        .def("initArbitraryPureState", &initArbitraryPureState)
        .def("initRandomPureState", &initRandomPureState)
        .def("initRandomMixedState", &initRandomMixedState)

        // Setters
        .def("setQuregAmps", (void (*)(Qureg, qindex, vector<qcomp>))(&setQuregAmps))
        .def("setDensityQuregAmps", (void (*)(Qureg, qindex, qindex, vector<vector<qcomp>>))(&setDensityQuregAmps))
        .def("setDensityQuregFlatAmps", (void (*)(Qureg, qindex, vector<qcomp>))(&setDensityQuregFlatAmps))
        .def("setQuregToClone", (&setQuregToClone))
        .def("setQuregToWeightedSum", (void (*)(Qureg, vector<qcomp>, vector<Qureg>))(&setQuregToWeightedSum))
        .def("setQuregToMixture", (void (*)(Qureg, vector<qreal>, vector<Qureg>))(&setQuregToMixture))
        .def("setQuregToRenormalized", (&setQuregToRenormalized))
        .def("setQuregToPauliStrSum", (&setQuregToPauliStrSum))
        .def("setQuregToPartialTrace", (void (*)(Qureg, Qureg, vector<int>))(&setQuregToPartialTrace))
        .def("setQuregToReducedDensityMatrix", (void (*)(Qureg, Qureg, vector<int>))(&setQuregToReducedDensityMatrix))

        // Getters
        .def("getQuregAmps", (std::vector<qcomp> (*)(Qureg, qindex, qindex))(&getQuregAmps))
        .def("getDensityQuregAmps",
             (std::vector<std::vector<qcomp>> (*)(Qureg, qindex, qindex, qindex, qindex))(&getDensityQuregAmps))
        .def("getQuregAmp", (&getQuregAmp))
        .def("getDensityQuregAmp", (&getDensityQuregAmp))

        // Calculations
        .def("calcExpecPauliStr", (&calcExpecPauliStr))
        .def("calcExpecPauliStrSum", (&calcExpecPauliStrSum))
        .def("calcExpecFullStateDiagMatr", (&calcExpecFullStateDiagMatr))
        .def("calcExpecFullStateDiagMatrPower", (&calcExpecFullStateDiagMatrPower))
        .def("calcProbOfBasisState", (&calcProbOfBasisState))
        .def("calcProbOfQubitOutcome", (&calcProbOfQubitOutcome))
        .def("calcProbOfMultiQubitOutcome", (qreal (*)(Qureg, vector<int>, vector<int>))(&calcProbOfMultiQubitOutcome))
        .def("calcProbsOfAllMultiQubitOutcomes",
             (vector<qreal> (*)(Qureg, vector<int>))(&calcProbsOfAllMultiQubitOutcomes))
        .def("calcTotalProb", (&calcTotalProb))
        .def("calcPurity", (&calcPurity))
        .def("calcFidelity", (&calcFidelity))
        .def("calcDistance", (&calcDistance))
        .def("calcPartialTrace", (Qureg (*)(Qureg, vector<int>))(&calcPartialTrace))
        .def("calcReducedDensityMatrix", (Qureg (*)(Qureg, vector<int>))(&calcReducedDensityMatrix))
        .def("calcInnerProduct", (&calcInnerProduct))
        .def("calcExpecNonHermitianPauliStrSum", (&calcExpecNonHermitianPauliStrSum))
        .def("calcExpecNonHermitianFullStateDiagMatr", (&calcExpecNonHermitianFullStateDiagMatr))
        .def("calcExpecNonHermitianFullStateDiagMatrPower", (&calcExpecNonHermitianFullStateDiagMatrPower))

        // Multiplication
        .def("leftapplyCompMatr1", (&leftapplyCompMatr1))
        .def("rightapplyCompMatr1", (&rightapplyCompMatr1))
        .def("leftapplyCompMatr2", (&leftapplyCompMatr2))
        .def("rightapplyCompMatr2", (&rightapplyCompMatr2))
        .def("leftapplyCompMatr", (void (*)(Qureg, vector<int>, CompMatr))(&leftapplyCompMatr))
        .def("rightapplyCompMatr", (void (*)(Qureg, vector<int>, CompMatr))(&rightapplyCompMatr))
        .def("leftapplyDiagMatr1", (&leftapplyDiagMatr1))
        .def("rightapplyDiagMatr1", (&rightapplyDiagMatr1))
        .def("leftapplyDiagMatr2", (&leftapplyDiagMatr2))
        .def("rightapplyDiagMatr2", (&rightapplyDiagMatr2))
        .def("leftapplyDiagMatr", (void (*)(Qureg, vector<int>, DiagMatr))(&leftapplyDiagMatr))
        .def("rightapplyDiagMatr", (void (*)(Qureg, vector<int>, DiagMatr))(&rightapplyDiagMatr))
        .def("leftapplyDiagMatrPower", (void (*)(Qureg, vector<int>, DiagMatr, qcomp))(&leftapplyDiagMatrPower))
        .def("rightapplyDiagMatrPower", (void (*)(Qureg, vector<int>, DiagMatr, qcomp))(&rightapplyDiagMatrPower))
        .def("leftapplyFullStateDiagMatr", (&leftapplyFullStateDiagMatr))
        .def("rightapplyFullStateDiagMatr", (&rightapplyFullStateDiagMatr))
        .def("leftapplyFullStateDiagMatrPower", (&leftapplyFullStateDiagMatrPower))
        .def("rightapplyFullStateDiagMatrPower", (&rightapplyFullStateDiagMatrPower))
        .def("leftapplySwap", (&leftapplySwap))
        .def("rightapplySwap", (&rightapplySwap))
        .def("leftapplyPauliX", (&leftapplyPauliX))
        .def("rightapplyPauliX", (&rightapplyPauliX))
        .def("leftapplyPauliY", (&leftapplyPauliY))
        .def("rightapplyPauliY", (&rightapplyPauliY))
        .def("leftapplyPauliZ", (&leftapplyPauliZ))
        .def("rightapplyPauliZ", (&rightapplyPauliZ))
        .def("leftapplyPauliStr", (&leftapplyPauliStr))
        .def("rightapplyPauliStr", (&rightapplyPauliStr))
        .def("leftapplyPauliStrSum", (&leftapplyPauliStrSum))
        .def("rightapplyPauliStrSum", (&rightapplyPauliStrSum))
        .def("leftapplyPauliGadget", (&leftapplyPauliGadget))
        .def("rightapplyPauliGadget", (&rightapplyPauliGadget))
        .def("leftapplyPhaseGadget", (void (*)(Qureg, vector<int>, qreal))(&leftapplyPhaseGadget))
        .def("rightapplyPhaseGadget", (void (*)(Qureg, vector<int>, qreal))(&rightapplyPhaseGadget))
        .def("leftapplyMultiQubitNot", (void (*)(Qureg, vector<int>))(&leftapplyMultiQubitNot))
        .def("rightapplyMultiQubitNot", (void (*)(Qureg, vector<int>))(&rightapplyMultiQubitNot))
        .def("leftapplyQubitProjector", (&leftapplyQubitProjector))
        .def("rightapplyQubitProjector", (&rightapplyQubitProjector))
        // .def("leftapplyMultiQubitProjector", (&leftapplyMultiQubitProjector))
        // .def("rightapplyMultiQubitProjector", (&rightapplyMultiQubitProjector))

        // Operations
        .def("applyCompMatr1", (&applyCompMatr1))
        .def("applyControlledCompMatr1", (&applyControlledCompMatr1))
        //
        .def("applyMultiControlledCompMatr1",
             (void (*)(Qureg, vector<int>, int, CompMatr1))(&applyMultiControlledCompMatr1))
        .def("applyMultiStateControlledCompMatr1",
             (void (*)(Qureg, vector<int>, vector<int>, int, CompMatr1))(&applyMultiStateControlledCompMatr1))
        .def("applyCompMatr2", (&applyCompMatr2))
        .def("applyControlledCompMatr2", (&applyControlledCompMatr2))
        .def("applyMultiControlledCompMatr2",
             (void (*)(Qureg, vector<int>, int, int, CompMatr2))(&applyMultiControlledCompMatr2))
        .def("applyMultiStateControlledCompMatr2",
             (void (*)(Qureg, vector<int>, vector<int>, int, int, int, CompMatr2))(&applyMultiStateControlledCompMatr2))

        .def("applyCompMatr", (void (*)(Qureg, vector<int>, CompMatr))(&applyCompMatr))
        .def("applyControlledCompMatr", (void (*)(Qureg, int, vector<int>, CompMatr))(&applyControlledCompMatr))
        .def("applyMultiControlledCompMatr",
             (void (*)(Qureg, vector<int>, vector<int>, CompMatr))(&applyMultiControlledCompMatr))
        .def("applyMultiStateControlledCompMatr",
             (void (*)(Qureg, vector<int>, vector<int>, vector<int>, CompMatr))(&applyMultiStateControlledCompMatr))

        .def("applyDiagMatr1", (&applyDiagMatr1))
        .def("applyControlledDiagMatr1", (&applyControlledDiagMatr1))
        .def("applyMultiControlledDiagMatr1",
             (void (*)(Qureg, vector<int>, int, DiagMatr1))(&applyMultiControlledDiagMatr1))
        .def("applyMultiStateControlledDiagMatr1",
             (void (*)(Qureg, vector<int>, vector<int>, int, DiagMatr1))(&applyMultiStateControlledDiagMatr1))

        .def("applyDiagMatr2", (&applyDiagMatr2))
        .def("applyControlledDiagMatr2", (&applyControlledDiagMatr2))

        .def("applyMultiControlledDiagMatr2",
             (void (*)(Qureg, vector<int>, int, int, DiagMatr2))(&applyMultiControlledDiagMatr2))
        .def("applyMultiStateControlledDiagMatr2",
             (void (*)(Qureg, vector<int>, vector<int>, int, int, DiagMatr2))(&applyMultiStateControlledDiagMatr2))

        .def("applyDiagMatr", (void (*)(Qureg, vector<int>, DiagMatr))(&applyDiagMatr))
        .def("applyControlledDiagMatr", (void (*)(Qureg, int, vector<int>, DiagMatr))(&applyControlledDiagMatr))
        .def("applyMultiControlledDiagMatr",
             (void (*)(Qureg, vector<int>, vector<int>, DiagMatr))(&applyMultiControlledDiagMatr))
        .def("applyMultiStateControlledDiagMatr",
             (void (*)(Qureg, vector<int>, vector<int>, vector<int>, DiagMatr))(&applyMultiStateControlledDiagMatr))
        .def("applyDiagMatrPower", (void (*)(Qureg, vector<int>, DiagMatr, qcomp))(&applyDiagMatrPower))
        .def("applyControlledDiagMatrPower",
             (void (*)(Qureg, int, vector<int>, DiagMatr, qcomp))(&applyControlledDiagMatrPower))
        .def("applyMultiControlledDiagMatrPower",
             (void (*)(Qureg, vector<int>, vector<int>, DiagMatr, qcomp))(&applyMultiControlledDiagMatrPower))
        .def("applyMultiStateControlledDiagMatrPower", (void (*)(Qureg, vector<int>, vector<int>, vector<int>, DiagMatr,
                                                                 qcomp))(&applyMultiStateControlledDiagMatrPower))

        .def("applyFullStateDiagMatr", (&applyFullStateDiagMatr))
        .def("applyFullStateDiagMatrPower", (&applyFullStateDiagMatrPower))

        .def("applyS", (&applyS))
        .def("applyControlledS", (&applyControlledS))
        .def("applyMultiControlledS", (void (*)(Qureg, vector<int>, int))(&applyMultiControlledS))

        .def("applyT", (&applyT))
        .def("applyControlledT", (&applyControlledT))
        .def("applyMultiControlledT", (void (*)(Qureg, vector<int>, int))(&applyMultiControlledT))
        .def("applyMultiStateControlledT",
             (void (*)(Qureg, vector<int>, vector<int>, int))(&applyMultiStateControlledT))

        .def("applyHadamard", (&applyHadamard))
        .def("applyControlledHadamard", (&applyControlledHadamard))
        .def("applyMultiControlledHadamard", (void (*)(Qureg, vector<int>, int))(&applyMultiControlledHadamard))
        .def("applyMultiStateControlledHadamard",
             (void (*)(Qureg, vector<int>, vector<int>, int))(&applyMultiStateControlledHadamard))

        .def("applySwap", (&applySwap))
        .def("applyControlledSwap", (&applyControlledSwap))
        .def("applyMultiControlledSwap", (void (*)(Qureg, vector<int>, int, int))(&applyMultiControlledSwap))
        .def("applyMultiStateControlledSwap",
             (void (*)(Qureg, vector<int>, vector<int>, int, int))(&applyMultiStateControlledSwap))

        .def("applySqrtSwap", (&applySqrtSwap))
        .def("applyControlledSqrtSwap", (&applyControlledSqrtSwap))
        .def("applyMultiControlledSqrtSwap", (void (*)(Qureg, vector<int>, int, int))(&applyMultiControlledSqrtSwap))
        // .def("applyMultiStateControlledSqrtSwap", (void(*)(Qureg, vector<int>, vector<int>, int, int,
        // int))(&applyMultiStateControlledSqrtSwap))

        .def("applyPauliX", (&applyPauliX))
        .def("applyPauliY", (&applyPauliY))
        .def("applyPauliZ", (&applyPauliZ))
        .def("applyControlledPauliX", (&applyControlledPauliX))
        .def("applyControlledPauliY", (&applyControlledPauliY))
        .def("applyControlledPauliZ", (&applyControlledPauliZ))

        .def("applyMultiControlledPauliX", (void (*)(Qureg, vector<int>, int))(&applyMultiControlledPauliX))
        .def("applyMultiControlledPauliY", (void (*)(Qureg, vector<int>, int))(&applyMultiControlledPauliY))
        .def("applyMultiControlledPauliZ", (void (*)(Qureg, vector<int>, int))(&applyMultiControlledPauliZ))
        .def("applyMultiStateControlledPauliX",
             (void (*)(Qureg, vector<int>, vector<int>, int))(&applyMultiStateControlledPauliX))
        .def("applyMultiStateControlledPauliY",
             (void (*)(Qureg, vector<int>, vector<int>, int))(&applyMultiStateControlledPauliY))
        .def("applyMultiStateControlledPauliZ",
             (void (*)(Qureg, vector<int>, vector<int>, int))(&applyMultiStateControlledPauliZ))

        .def("applyPauliStr", (&applyPauliStr))
        .def("applyControlledPauliStr", (&applyControlledPauliStr))
        .def("applyMultiControlledPauliStr", (void (*)(Qureg, vector<int>, PauliStr))(&applyMultiControlledPauliStr))
        .def("applyMultiStateControlledPauliStr",
             (void (*)(Qureg, vector<int>, vector<int>, PauliStr))(&applyMultiStateControlledPauliStr))

        .def("applyRotateX", (&applyRotateX))
        .def("applyRotateY", (&applyRotateY))
        .def("applyRotateZ", (&applyRotateZ))
        .def("applyControlledRotateX", (&applyControlledRotateX))
        .def("applyControlledRotateY", (&applyControlledRotateY))
        .def("applyControlledRotateZ", (&applyControlledRotateZ))

        .def("applyMultiControlledRotateX", (void (*)(Qureg, vector<int>, int, qreal))(&applyMultiControlledRotateX))
        .def("applyMultiControlledRotateY", (void (*)(Qureg, vector<int>, int, qreal))(&applyMultiControlledRotateY))
        .def("applyMultiControlledRotateZ", (void (*)(Qureg, vector<int>, int, qreal))(&applyMultiControlledRotateZ))
        .def("applyMultiStateControlledRotateX",
             (void (*)(Qureg, vector<int>, vector<int>, int, qreal))(&applyMultiStateControlledRotateX))
        .def("applyMultiStateControlledRotateY",
             (void (*)(Qureg, vector<int>, vector<int>, int, qreal))(&applyMultiStateControlledRotateY))
        .def("applyMultiStateControlledRotateZ",
             (void (*)(Qureg, vector<int>, vector<int>, int, qreal))(&applyMultiStateControlledRotateZ))

        .def("applyRotateAroundAxis", (&applyRotateAroundAxis))
        .def("applyControlledRotateAroundAxis", (&applyControlledRotateAroundAxis))
        .def("applyMultiControlledRotateAroundAxis",
             (void (*)(Qureg, vector<int>, int, qreal, qreal, qreal, qreal))(&applyMultiControlledRotateAroundAxis))
        .def("applyMultiStateControlledRotateAroundAxis",
             (void (*)(Qureg, vector<int>, vector<int>, int, qreal, qreal, qreal, qreal))(
                 &applyMultiStateControlledRotateAroundAxis))

        .def("applyPauliGadget", (&applyPauliGadget))
        .def("applyNonUnitaryPauliGadget", (&applyNonUnitaryPauliGadget))
        .def("applyControlledPauliGadget", (&applyControlledPauliGadget))

        .def("applyMultiControlledPauliGadget",
             (void (*)(Qureg, vector<int>, PauliStr, qreal))(&applyMultiControlledPauliGadget))
        .def("applyMultiStateControlledPauliGadget",
             (void (*)(Qureg, vector<int>, vector<int>, PauliStr, qreal))(&applyMultiStateControlledPauliGadget))

        .def("applyPhaseGadget", (void (*)(Qureg, vector<int>, qreal))(&applyPhaseGadget))
        .def("applyControlledPhaseGadget", (void (*)(Qureg, int, vector<int>, qreal))(&applyControlledPhaseGadget))

        .def("applyMultiControlledPhaseGadget",
             (void (*)(Qureg, vector<int>, vector<int>, qreal))(&applyMultiControlledPhaseGadget))
        .def("applyMultiStateControlledPhaseGadget",
             (void (*)(Qureg, vector<int>, vector<int>, vector<int>, qreal))(&applyMultiStateControlledPhaseGadget))

        .def("applyPhaseFlip", (&applyPhaseFlip))
        .def("applyPhaseShift", (&applyPhaseShift))
        .def("applyTwoQubitPhaseFlip", (&applyTwoQubitPhaseFlip))
        .def("applyTwoQubitPhaseShift", (&applyTwoQubitPhaseShift))

        .def("applyMultiQubitPhaseFlip", (void (*)(Qureg, vector<int>))(&applyMultiQubitPhaseFlip))
        .def("applyMultiQubitPhaseShift", (void (*)(Qureg, vector<int>, qreal))(&applyMultiQubitPhaseShift))
        .def("applyMultiQubitNot", (void (*)(Qureg, vector<int>))(&applyMultiQubitNot))
        .def("applyControlledMultiQubitNot", (void (*)(Qureg, int, vector<int>))(&applyControlledMultiQubitNot))
        .def("applyMultiControlledMultiQubitNot",
             (void (*)(Qureg, vector<int>, vector<int>))(&applyMultiControlledMultiQubitNot))
        .def("applyMultiStateControlledMultiQubitNot",
             (void (*)(Qureg, vector<int>, vector<int>, vector<int>))(&applyMultiStateControlledMultiQubitNot))

        .def("applyQubitMeasurement", (&applyQubitMeasurement))
        .def("applyForcedQubitMeasurement", (&applyForcedQubitMeasurement))
        // applyMultiQubitMeasurementAndGetProb
        .def("applyForcedMultiQubitMeasurement",
             (qreal (*)(Qureg, vector<int>, vector<int>))(&applyForcedMultiQubitMeasurement))

        .def("applyQubitProjector", (&applyQubitProjector))
        .def("applyMultiQubitProjector", (void (*)(Qureg, vector<int>, vector<int>))(&applyMultiQubitProjector))

        .def("applyQuantumFourierTransform", (void (*)(Qureg, vector<int>))(&applyQuantumFourierTransform))
        .def("applyFullQuantumFourierTransform", (&applyFullQuantumFourierTransform))

        // Decoherence
        .def("mixDephasing", (&mixDephasing))
        .def("mixTwoQubitDephasing", (&mixTwoQubitDephasing))
        .def("mixDepolarising", (&mixDepolarising))
        .def("mixTwoQubitDepolarising", (&mixTwoQubitDepolarising))
        .def("mixDamping", (&mixDamping))
        .def("mixPaulis", (&mixPaulis))
        .def("mixQureg", (&mixQureg))
        .def("mixKrausMap", (void (*)(Qureg, vector<int>, KrausMap))(&mixKrausMap))
        .def("mixSuperOp", (void (*)(Qureg, vector<int>, SuperOp))(&mixSuperOp))

        // Trotterisation
        .def("applyTrotterizedPauliStrSumGadget", (&applyTrotterizedPauliStrSumGadget))
        .def("applyTrotterizedControlledPauliStrSumGadget", (&applyTrotterizedControlledPauliStrSumGadget))
        .def("applyTrotterizedMultiControlledPauliStrSumGadget",
             (void (*)(Qureg, vector<int>, PauliStrSum, qreal, int, int))(
                 &applyTrotterizedMultiControlledPauliStrSumGadget))
        .def("applyTrotterizedMultiStateControlledPauliStrSumGadget",
             (void (*)(Qureg, vector<int>, vector<int>, PauliStrSum, qreal, int, int))(
                 &applyTrotterizedMultiStateControlledPauliStrSumGadget))
        .def("applyTrotterizedNonUnitaryPauliStrSumGadget", (&applyTrotterizedNonUnitaryPauliStrSumGadget))
        .def("applyTrotterizedUnitaryTimeEvolution", (&applyTrotterizedUnitaryTimeEvolution))
        .def("applyTrotterizedImaginaryTimeEvolution", (&applyTrotterizedImaginaryTimeEvolution))
        .def("applyTrotterizedNoisyTimeEvolution", (&applyTrotterizedNoisyTimeEvolution))

        // Syncing
        .def("syncQuregToGpu", (&syncQuregToGpu))
        .def("syncQuregFromGpu", (&syncQuregFromGpu))
        .def("syncSubQuregToGpu", (&syncSubQuregToGpu))
        .def("syncSubQuregFromGpu", (&syncSubQuregFromGpu))

        // Reporters
        .def("reportQuregParams", (&reportQuregParams))
        .def("report", (&reportQureg));

    // KrausMap
    nb::class_<KrausMap>(quest, "KrausMap")
        .def(nb::new_(&createKrausMap))
        .def(nb::new_(&createInlineKrausMap))
        .def("set", (void (*)(KrausMap, vector<vector<vector<qcomp>>>))(&setKrausMap))
        .def("__del__", &destroyKrausMap)
        .def("sync", (&syncKrausMap))
        .def("report", (&reportKrausMap));

    // SuperOp
    nb::class_<SuperOp>(quest, "SuperOp")
        .def(nb::new_(&createSuperOp))
        .def(nb::new_(&createInlineSuperOp))
        .def("set", (void (*)(SuperOp, vector<vector<qcomp>>))(&setSuperOp))
        .def("__del__", &destroySuperOp)
        .def("sync", (&syncSuperOp))
        .def("report", (&reportSuperOp));

    // CompMatr1
    nb::class_<CompMatr1>(quest, "CompMatr1")
        .def(nb::new_((CompMatr1 (*)(vector<vector<qcomp>>))(&getCompMatr1)))
        .def("report", (&reportCompMatr1));

    // CompMatr2
    nb::class_<CompMatr2>(quest, "CompMatr2")
        .def(nb::new_((CompMatr2 (*)(vector<vector<qcomp>>))(&getCompMatr2)))
        .def("report", (&reportCompMatr2));

    // CompMatr
    nb::class_<CompMatr>(quest, "CompMatr")
        .def(nb::new_(&createCompMatr))
        .def("set", (void (*)(CompMatr, vector<vector<qcomp>>))(&setCompMatr))
        .def("__del__", &destroyCompMatr)
        .def("sync", (&syncCompMatr))
        .def("report", (&reportCompMatr));

    // DiagMatr1
    nb::class_<DiagMatr1>(quest, "DiagMatr1")
        .def(nb::new_((DiagMatr1 (*)(vector<qcomp>))(&getDiagMatr1)))
        .def("report", (&reportDiagMatr1));

    // DiagMatr2
    nb::class_<DiagMatr2>(quest, "DiagMatr2")
        .def(nb::new_((DiagMatr2 (*)(vector<qcomp>))(&getDiagMatr2)))
        .def("report", (&reportDiagMatr2));

    // DiagMatr
    nb::class_<DiagMatr>(quest, "DiagMatr")
        .def(nb::new_(&createDiagMatr))
        .def("set", (void (*)(DiagMatr, vector<qcomp>))(&setDiagMatr))
        .def("__del__", &destroyDiagMatr)
        .def("sync", (&syncDiagMatr))
        .def("report", (&reportDiagMatr));

    // FullStateDiagMatr
    nb::class_<FullStateDiagMatr>(quest, "FullStateDiagMatr")
        .def(nb::new_(&createFullStateDiagMatr))
        .def("set", (void (*)(FullStateDiagMatr, qindex, vector<qcomp>))(&setFullStateDiagMatr))
        .def("__del__", &destroyFullStateDiagMatr)
        .def("sync", (&syncFullStateDiagMatr))
        .def("report", (&reportFullStateDiagMatr));
}
